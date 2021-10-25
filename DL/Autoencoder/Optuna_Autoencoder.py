import optuna
from optuna.trial import TrialState

def objective(trial):
    """Objective function to be optimized by Optuna.
    Hyperparameters chosen to be optimized: optimizer, learning rate,
    dropout values, number of convolutional layers, number of filters of
    convolutional layers, number of neurons of fully connected layers.
    Inputs:
        - trial (optuna.trial._trial.Trial): Optuna trial
    Returns:
        - accuracy(torch.Tensor): The test accuracy. Parameter to be maximized.
    """

    # Define range of values to be tested for the hyperparameters
    n_enc_layers = trial.suggest_int("n_enc_layers", 1, 5)  # Number of encoder layers

    # Make an intial guess at the number of nodes in the first layer
    enc_node_list = [trial.suggest_int('enc_layer_nodes_0', 256, 1024)]
    n_latent_dims = trial.suggest_int('n_latent_dims', 8, 32)

    for enc_layer_n in range(1, n_enc_layers-1):

        # Suggest the number of nodes for the next layer
        # can't be greater than the number in the previous layer

        n_nodes = trial.suggest_int(f'enc_layer_nodes_{enc_layer_n}'
        , enc_node_list[0] - n_latent_dims/n_enc_layers
        , enc_node_list[-1])
        
        enc_node_list.append(n_nodes)



    


    # Define range of values to be tested for the hyperparameters
    n_dec_layers = trial.suggest_int("n_dec_layers", 1, 5)  # Number of encoder layers

    # Make an intial guess at the number of nodes in the first layer
    enc_node_list = [trial.suggest_int('dec_layer_nodes_0', 256, 1024)]

    for enc_layer_n in range(1, n_enc_layers-1):

        # Suggest the number of nodes for the 
        n_nodes = trial.suggest_int(f'enc_layer_nodes_{enc_layer_n}')
        enc_node_list.append(n_nodes)



    # Generate the model
    model = Autoencoder(trial, latent_dim, n_enc_layers, enc_reductions, n_dec_layers, dec_expansions)

    # Generate the optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])  # Optimizers
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)                                 # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # Training of the model
    for epoch in range(n_epochs):
        train(model, optimizer)  # Train the model
        accuracy = test(model)   # Evaluate the model

        # For pruning (stops trial early if not promising)
        trial.report(accuracy, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy