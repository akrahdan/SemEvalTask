from __future__ import absolute_import, division, print_function
from audioop import mul
import collections
import logging
import math
import os
import random
import warnings
from dataclasses import asdict
from multiprocessing import cpu_count
import tempfile
from pathlib import Path

from collections import Counter
import numpy as np
import pandas as pd
import torch
from scipy.stats import mode, pearsonr
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    mean_squared_error,
    roc_curve,
    auc,
    average_precision_score,
)

# torch

from torch.utils.tensorboard import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from semevaltasks.losses.loss_utils import init_loss
from semevaltasks.classification.utils import (
    InputExample,
    ClassificationDataset,
    LazyClassificationDataset,
    convert_example_to_feature,
    flatten_results
)
from tqdm.auto import tqdm, trange
from transformers.optimization import AdamW, Adafactor
from config import SemEvalModelArgs
import os
from transformers import (
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForSequenceClassification,
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForSequenceClassification
)
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

class SemevalModel:
    def __init__(
        self,
        model_type,
        model_name,
        tokenizer_type=None,
        tokenizer_name=None,
        num_labels=None,
        weight=None,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        **kwargs,):

        MODEL_CLASSES = {
            "electra": (
                ElectraConfig,
                ElectraForSequenceClassification,
                ElectraTokenizerFast,
            ),
  
            "roberta": (
                RobertaConfig,
                RobertaForSequenceClassification,
                RobertaTokenizerFast,
            )
         
        }
        self.args = self._load_model_args(model_name)

        if isinstance(args, SemEvalModelArgs):
            self.args = args
        
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)
            
        len_labels_list = 2 if not num_labels else num_labels
        self.args.labels_list = [i for i in range(len_labels_list)]

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

        if tokenizer_type is not None:
            if isinstance(tokenizer_type, str):
                _, _, tokenizer_class = MODEL_CLASSES[tokenizer_type]
            else:
                tokenizer_class = tokenizer_type

        if num_labels:
            self.config = config_class.from_pretrained(
                model_name, num_labels=num_labels, **self.args.config
            )
            self.num_labels = num_labels
        else:
            self.config = config_class.from_pretrained(model_name, **self.args.config)
            self.num_labels = self.config.num_labels
        
        self.weight = weight


        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"
        
        self.loss_fct = init_loss(
            weight=self.weight, device=self.device, args=self.args
        )

        if not self.args.quantized_model:
            self.model = model_class.from_pretrained(
                model_name, config=self.config, **kwargs
            )
        else:
            quantized_weights = torch.load(
                os.path.join(model_name, "pytorch_model.bin")
            )

            self.model = model_class.from_pretrained(
                None, config=self.config, state_dict=quantized_weights
            )

        if self.args.dynamic_quantize:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if self.args.quantized_model:
            self.model.load_state_dict(quantized_weights)
        if self.args.dynamic_quantize:
            self.args.quantized_model = True
        
        self.results = {}

        if tokenizer_name is None:
            tokenizer_name = model_name
        
        self.tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name, do_lower_case=self.args.do_lower_case, **kwargs
        )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(
                self.args.special_tokens_list, special_tokens=True
            )
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type
        self.args.tokenizer_name = tokenizer_name
        self.args.tokenizer_type = tokenizer_type



        

            
        



            

    def train_model(self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,):
        """
        Train models using 'train_df'

        Args:
        
        Returns:
           global_step: Number of global steps trained
           training_details: Average training loss
        """

        if args:
            self.args.update_from_dict(args)
        
        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )
        if not output_dir:
            output_dir = self.args.output_dir
        
        if (
            os.path.exists(output_dir)
            and os.listdir(output_dir)
            and not self.args.overwrite_output_dir
            ):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(
                    output_dir
                )
            )
        self._move_model_to_device()

        if "text" in train_df.columns and "labels" in train_df.columns:
            train_examples = (
                train_df["text"].astype(str).tolist(),
                train_df["labels"].tolist()
            )
        elif "text_a" in train_df.columns and "text_b" in train_df.columns:
            train_examples = (
                train_df["text_a"].astype(str).tolist(),
                train_df["text_b"].astype(str).tolist(),
                train_df["labels"].tolist(),
            )
        else:
            warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
            train_examples = (
                train_df.iloc[:, 0].astype(str).tolist(),
                train_df.iloc[:, 1].tolist(),
            )
        
        train_dataset = self.load_and_cache_examples(
            train_examples, verbose=verbose
        )

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers
        )
        os.makedirs(output_dir, exist_ok=True)
        global_step, training_details = self.train(
            train_dataloader,
            output_dir,
            multi_label,
            show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(model=self.model)

        if verbose:
            logger.info(
                " Training of {} model complete. Saved to {}.".format(
                    self.args.model_type, output_dir
                )
            )
        return global_step, training_details

        

    def evaluate(
        self,
        eval_df,
        output_dir,
        multi_label=False,
        prefix="",
        verbose=True,
        silent=False,
        wandb_log=True,
        **kwargs,
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}
        
        if isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError(
                    "Lazy loading is not implemented for LayoutLM models"
                )
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError(
                    "Input must be given as a path to a file when using lazy loading"
                )

            if "text" in eval_df.columns and "labels" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    eval_examples = [
                        InputExample(i, text, None, label, x0, y0, x1, y1)
                        for i, (text, label, x0, y0, x1, y1) in enumerate(
                            zip(
                                eval_df["text"].astype(str),
                                eval_df["labels"],
                                eval_df["x0"],
                                eval_df["y0"],
                                eval_df["x1"],
                                eval_df["y1"],
                            )
                        )
                    ]
                else:
                    eval_examples = (
                        eval_df["text"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                    )
            elif "text_a" in eval_df.columns and "text_b" in eval_df.columns:
                if self.args.model_type == "layoutlm":
                    raise ValueError("LayoutLM cannot be used with sentence-pair tasks")
                else:
                    eval_examples = (
                        eval_df["text_a"].astype(str).tolist(),
                        eval_df["text_b"].astype(str).tolist(),
                        eval_df["labels"].tolist(),
                    )
            else:
                warnings.warn(
                    "Dataframe headers not specified. Falling back to using column 0 as text and column 1 as labels."
                )
                eval_examples = (
                    eval_df.iloc[:, 0].astype(str).tolist(),
                    eval_df.iloc[:, 1].tolist(),
                )

            if args.sliding_window:
                eval_dataset, window_counts = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
            else:
                eval_dataset = self.load_and_cache_examples(
                    eval_examples, evaluate=True, verbose=verbose, silent=silent
                )
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset), self.num_labels))
        if multi_label:
            out_label_ids = np.empty((len(eval_dataset), self.num_labels))
        else:
            out_label_ids = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(
            tqdm(
                eval_dataloader,
                disable=args.silent or silent,
                desc="Running Evaluation",
            )
        ):
            # batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                        tmp_eval_loss, logits = outputs[:2]
                else:
                    outputs = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )
                    tmp_eval_loss, logits = outputs[:2]

                if multi_label:
                    logits = logits.sigmoid()
                if self.args.n_gpu > 1:
                    tmp_eval_loss = tmp_eval_loss.mean()
                eval_loss += tmp_eval_loss.item()

            nb_eval_steps += 1

            start_index = self.args.eval_batch_size * i
            end_index = (
                start_index + self.args.eval_batch_size
                if i != (n_batches - 1)
                else len(eval_dataset)
            )
            preds[start_index:end_index] = logits.detach().cpu().numpy()
            out_label_ids[start_index:end_index] = (
                inputs["labels"].detach().cpu().numpy()
            )

            # if preds is None:
            #     preds = logits.detach().cpu().numpy()
            #     out_label_ids = inputs["labels"].detach().cpu().numpy()
            # else:
            #     preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #     out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.sliding_window:
            count = 0
            window_ranges = []
            for n_windows in window_counts:
                window_ranges.append([count, count + n_windows])
                count += n_windows

            preds = [
                preds[window_range[0] : window_range[1]]
                for window_range in window_ranges
            ]
            out_label_ids = [
                out_label_ids[i]
                for i in range(len(out_label_ids))
                if i in [window[0] for window in window_ranges]
            ]

            model_outputs = preds
            if args.regression is True:
                preds = [np.squeeze(pred) for pred in preds]
                final_preds = []
                for pred_row in preds:
                    mean_pred = np.mean(pred_row)
                    print(mean_pred)
                    final_preds.append(mean_pred)
            else:
                preds = [np.argmax(pred, axis=1) for pred in preds]
                final_preds = []
                for pred_row in preds:
                    val_freqs_desc = Counter(pred_row).most_common()
                    if (
                        len(val_freqs_desc) > 1
                        and val_freqs_desc[0][1] == val_freqs_desc[1][1]
                    ):
                        final_preds.append(args.tie_value)
                    else:
                        final_preds.append(val_freqs_desc[0][0])
            preds = np.array(final_preds)
        elif not multi_label and args.regression is True:
            preds = np.squeeze(preds)
            model_outputs = preds
        else:
            model_outputs = preds

            if not multi_label:
                preds = np.argmax(preds, axis=1)

        result, wrong = self.compute_metrics(
            preds, model_outputs, out_label_ids, eval_examples, **kwargs
        )
        result["eval_loss"] = eval_loss
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        if (
            self.args.wandb_project
            and wandb_log
            and not multi_label
            and not self.args.regression
        ):
            
            if not args.labels_map:
                self.args.labels_map = {i: i for i in range(self.num_labels)}

            labels_list = sorted(list(self.args.labels_map.keys()))
            inverse_labels_map = {
                value: key for key, value in self.args.labels_map.items()
            }

            truth = [inverse_labels_map[out] for out in out_label_ids]

        return results, model_outputs, wrong

    def train(
        self,
        train_dataloader,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        test_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(
                multi_label, **kwargs
            )

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        loss, *_ = self._calculate_loss(
                            model,
                            inputs,
                            loss_fct=self.loss_fct,
                            num_labels=self.num_labels,
                            args=self.args,
                        )
                else:
                    loss, *_ = self._calculate_loss(
                        model,
                        inputs,
                        loss_fct=self.loss_fct,
                        num_labels=self.num_labels,
                        args=self.args,
                    )

                if args.n_gpu > 1:
                    loss = (
                        loss.mean()
                    )  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "loss",
                            (tr_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                        logging_loss = tr_loss
                      

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        self.save_model(
                            output_dir_current, optimizer, scheduler, model=model
                        )

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ = self.eval_model(
                            eval_df,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )

                        output_dir_current = os.path.join(
                            output_dir, "checkpoint-{}".format(global_step)
                        )

                        if args.save_eval_checkpoints:
                            self.save_model(
                                output_dir_current,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])

                        if test_df is not None:
                            test_results, _, _ = self.eval_model(
                                test_df,
                                verbose=verbose
                                and args.evaluate_during_training_verbose,
                                silent=args.evaluate_during_training_silent,
                                wandb_log=False,
                                **kwargs,
                            )
                            for key in test_results:
                                training_progress_scores["test_" + key].append(
                                    test_results[key]
                                )

                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(
                                args.output_dir, "training_progress_scores.csv"
                            ),
                            index=False,
                        )

                      
                        for key, value in flatten_results(
                            self._get_last_metrics(training_progress_scores)
                        ).items():
                            try:
                                tb_writer.add_scalar(key, value, global_step)
                            except (NotImplementedError, AssertionError):
                                if verbose:
                                    logger.warning(
                                        f"can't log value of type: {type(value)} to tensorboar"
                                    )
                        tb_writer.flush()

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(
                                args.best_model_dir,
                                optimizer,
                                scheduler,
                                model=model,
                                results=results,
                            )
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if (
                                best_eval_metric - results[args.early_stopping_metric]
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if (
                                results[args.early_stopping_metric] - best_eval_metric
                                > args.early_stopping_delta
                            ):
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir,
                                    optimizer,
                                    scheduler,
                                    model=model,
                                    results=results,
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if (
                                        early_stopping_counter
                                        < args.early_stopping_patience
                                    ):
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(
                                                f" No improvement in {args.early_stopping_metric}"
                                            )
                                            logger.info(
                                                f" Current step: {early_stopping_counter}"
                                            )
                                            logger.info(
                                                f" Early stopping patience: {args.early_stopping_patience}"
                                            )
                                    else:
                                        if verbose:
                                            logger.info(
                                                f" Patience of {args.early_stopping_patience} steps reached"
                                            )
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results, _, _ = self.eval_model(
                    eval_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(
                    output_dir_current, optimizer, scheduler, results=results
                )

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                if test_df is not None:
                    test_results, _, _ = self.eval_model(
                        test_df,
                        verbose=verbose and args.evaluate_during_training_verbose,
                        silent=args.evaluate_during_training_silent,
                        wandb_log=False,
                        **kwargs,
                    )
                    for key in test_results:
                        training_progress_scores["test_" + key].append(
                            test_results[key]
                        )

                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(args.output_dir, "training_progress_scores.csv"),
                    index=False,
                )

               
                for key, value in flatten_results(
                    self._get_last_metrics(training_progress_scores)
                ).items():
                    try:
                        tb_writer.add_scalar(key, value, global_step)
                    except (NotImplementedError, AssertionError):
                        if verbose:
                            logger.warning(
                                f"can't log value of type: {type(value)} to tensorboar"
                            )
                tb_writer.flush()

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(
                        args.best_model_dir,
                        optimizer,
                        scheduler,
                        model=model,
                        results=results,
                    )
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if (
                        best_eval_metric - results[args.early_stopping_metric]
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if (
                        results[args.early_stopping_metric] - best_eval_metric
                        > args.early_stopping_delta
                    ):
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(
                            args.best_model_dir,
                            optimizer,
                            scheduler,
                            model=model,
                            results=results,
                        )
                        early_stopping_counter = 0
                    else:
                        if (
                            args.use_early_stopping
                            and args.early_stopping_consider_epochs
                        ):
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(
                                        f" No improvement in {args.early_stopping_metric}"
                                    )
                                    logger.info(
                                        f" Current step: {early_stopping_counter}"
                                    )
                                    logger.info(
                                        f" Early stopping patience: {args.early_stopping_patience}"
                                    )
                            else:
                                if verbose:
                                    logger.info(
                                        f" Patience of {args.early_stopping_patience} steps reached"
                                    )
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def load_and_cache_examples(
        self,
        examples,
        evaluate=False,
        no_cache=False,
        multi_label=False,
        verbose=True,
        silent=False,
        ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """
        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache
        
        if not multi_label and args.regression:
            output_mode = "regression"
        else:
            output_mode = "classification"
        
        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)
        mode = "dev" if evaluate else "train"
        dataset = ClassificationDataset(
            examples,
            self.tokenizer,
            self.args,
            mode=mode,
            multi_label=multi_label,
            output_mode=output_mode,
            no_cache=no_cache
        )
        return dataset




    def _load_model_args(self, input_args):
        args = SemEvalModelArgs()
        args.load(input_args)
        return args
    

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _move_model_to_device(self):
        self.model.to(self.device)
    
    def save_model(
        self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None
        ):
        if not output_dir:
          output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(
                    optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                )
                torch.save(
                    scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                )
            self.save_model_args(output_dir)
        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))
        
        
