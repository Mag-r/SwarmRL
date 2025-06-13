from flax.training import train_state


class CustomTrainState(train_state.TrainState):
    batch_stats: any = None


