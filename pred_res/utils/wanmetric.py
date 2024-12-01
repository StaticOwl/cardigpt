import wandb
from datetime import datetime

class WandbLogger:
    def __init__(self, project_name, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run = wandb.init(project=project_name, name=run_name, reinit=True)

    def log_table(self, name, dataframe):
        table = wandb.Table(dataframe=dataframe)
        self.run.log({name: table})

    def log_conf_matrix(self, y_true, y_pred, class_name):
        self.run.log({
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_name
            )
        })
    
    def log_metric(self, name, value, step=None):
        if step is not None:
            self.run.log({name: value, "epoch": step})
        else:
            self.run.log({name: value})

    def log_metrics(self, metrics, step=None):
        if step is not None:
            metrics["epoch"] = step
        self.run.log(metrics)
    
    def log_model(self, model_name, epoch):
        self.run.log({f"Model Checkpoint (Epoch {epoch})": wandb.Artifact(model_name, type="model")})

    def close(self):
        self.run.finish()