import os
from datetime import datetime
from pytorch_lightning.callbacks import Callback


class CustomCheckpointCallback(Callback):
    def __init__(self, dirpath, save_name, verbose=True, min_delta=0.0, save_top_k=1):
        super().__init__()
        self.best_metrics = []  # List of tuples (val_loss, val_pearson, file_path)
        self.best_pearson = float('-inf')
        self.best_loss = float('inf')
        self.dirpath = dirpath
        self.save_name = save_name
        self.verbose = verbose
        self.min_delta = min_delta
        self.save_top_k = save_top_k

    def on_validation_epoch_end(self, trainer, pl_module):
        current_pearson = trainer.callback_metrics.get('val_pearson')
        current_loss = trainer.callback_metrics.get('val_loss')
        epoch = trainer.current_epoch
        step = trainer.global_step

        if self.verbose:
            print(f"\nEpoch {epoch}, Step {step}: val_loss = {current_loss:.3f}, val_pearson = {current_pearson:.3f}")

        save_path = None
        if current_pearson is not None and current_loss is not None:
            if current_pearson > self.best_pearson + self.min_delta and current_loss < self.best_loss - self.min_delta:
                pearson_improvement = current_pearson - self.best_pearson
                loss_improvement = self.best_loss - current_loss
                self.best_pearson = current_pearson
                self.best_loss = current_loss
                save_path = self._get_file_path(epoch, step, current_pearson, current_loss)
                trainer.save_checkpoint(save_path)
                self.best_metrics.append((current_loss, current_pearson, save_path)) # latest element is the best
                self.best_metrics.sort(key=lambda x: (-x[0], x[1])) # descending loss, ascending pearson
                top_idx = 0
                for idx in range(-1, -len(self.best_metrics)-1, -1):
                    if self.best_metrics[idx][2] == save_path:
                        top_idx = idx
                        break
                if self.verbose:
                    print(f"\nEpoch {epoch}, global step {step}: 'val_pearson' improved by {pearson_improvement:.3f}. New best pearson: {current_pearson:.5f}")
                    print(f"Epoch {epoch}, global step {step}: 'val_loss' improved by {loss_improvement:.3f}. New best loss: {current_loss:.5f}")
                if len(self.best_metrics) > self.save_top_k:
                    _, _, path_to_remove = self.best_metrics.pop(0)
                    if os.path.exists(path_to_remove):
                        if self.verbose:
                            print(f"Removing checkpoint '{path_to_remove}'")
                        os.remove(path_to_remove)
                if self.verbose:
                    print(f"Both metrics improved, saving model to '{save_path}' as top {-top_idx} out of {len(self.best_metrics)} checkpoints.\n")
            
            elif current_pearson > self.best_pearson + self.min_delta:
                pearson_improvement = current_pearson - self.best_pearson
                self.best_pearson = current_pearson
                if self.verbose:
                    print(f"\nEpoch {epoch}, global step {step}: 'val_pearson' improved by {pearson_improvement:.3f} >= min_delta = {self.min_delta}. New best pearson: {current_pearson:.5f}\n")
            
            elif current_loss < self.best_loss - self.min_delta:
                loss_improvement = self.best_loss - current_loss
                self.best_loss = current_loss
                if self.verbose:
                    print(f"\nEpoch {epoch}, global step {step}: 'val_loss' improved by {loss_improvement:.3f} >= min_delta = {self.min_delta}. New best loss: {current_loss:.5f}\n")
            
            else:
                if self.verbose:
                    print(f"\nEpoch {epoch}, global step {step}: Neither metric improved, skipping checkpointing.\n")
                pass

    def _get_file_path(self, epoch, step, pearson, loss):
        return os.path.join(
            self.dirpath,
            f"{self.save_name}_{epoch:03d}_{step:05d}_{pearson:.3f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt"
        )



class CustomEarlyStoppingCallback(Callback):
    def __init__(self, patience=5, common=True, verbose=True):
        super().__init__()
        self.common = common
        self.patience = patience
        self.verbose = verbose
        self.best_pearson = float('-inf')
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.epochs_no_improve_pearson = 0
        self.epochs_no_improve_loss = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        current_pearson = trainer.callback_metrics.get('val_pearson')
        current_loss = trainer.callback_metrics.get('val_loss')
        
        if self.common:
            if current_pearson is not None and current_loss is not None:
                if current_pearson > self.best_pearson and current_loss > self.best_loss:
                    self.best_pearson = current_pearson
                    self.best_loss = current_loss
                    self.epochs_no_improve_pearson = 0
                else:
                    self.epochs_no_improve += 1

            # Check if early stopping is triggered
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"\nEarlyStopping triggered: botth val_pearson and val_loss has not improved for {self.epochs_no_improve} epochs.\n")
                trainer.should_stop = True
        else:
            # Check for improvement in val_pearson
            if current_pearson is not None:
                if current_pearson > self.best_pearson:
                    self.best_pearson = current_pearson
                    self.epochs_no_improve_pearson = 0
                else:
                    self.epochs_no_improve_pearson += 1

            # Check for improvement in val_loss
            if current_loss is not None:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.epochs_no_improve_loss = 0
                else:
                    self.epochs_no_improve_loss += 1

            # Check if early stopping is triggered
            if self.epochs_no_improve_pearson >= self.patience or self.epochs_no_improve_loss >= self.patience:
                if self.verbose:
                    print(f'EarlyStopping triggered: val_pearson has not improved for {self.epochs_no_improve_pearson} epochs and val_loss for {self.epochs_no_improve_loss} epochs.')
                trainer.should_stop = True