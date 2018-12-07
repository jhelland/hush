from imports import * 


# ROC-AUC estimate
class RocAucEvaluation(Callback):
    """
    AUROC: Given a randomly selected member $x$ of class $C_j$, AUROC estimates the
           probability that the model classifies $x$ to $C_j$.
    """
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC | epoch: {} | score: {:.6f} \n".format(epoch+1, score))


# for the sake of plotting
class LossHistory(keras.callbacks.Callback):
    """
    Tracks losses over epochs in a list.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))


class SlackMessenger:
    """
    Class wrapper that gives a convenient way to send a slack message to a channel of choice.
    Requires a generated key for the channel, stored in ../config.cfg.
    """
    def __init__(self, channel="general"):
        conf = configparser.ConfigParser()
        conf.read("../config.cfg")
        SLACK_TOKEN = conf.get("Slack", "token")
        self.sc = SlackClient(SLACK_TOKEN)
        self.channel = channel

    def message(self, msg):
        self.sc.api_call(
            "chat.postMessage",
            link_names=True,
            channel=self.channel,
            text=msg
        )
