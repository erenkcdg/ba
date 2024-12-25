def get_params(config, ds):
    if ds == 0:
        current_dataset = 'MNIST'
        num_epochs          = config['parameters_mnist']['num_epochs']
        batch_size          = config['parameters_mnist']['batch_size']
        learning_rate       = config['parameters_mnist']['learning_rate']
        general_seed        = config['parameters_mnist']['general_seed']
        num_trainings       = config['parameters_mnist']['num_trainings']
        length_dataset      = config['parameters_mnist']['length_dataset']
        outliers_indices    = config['parameters_mnist']['outliers_indices']
        current_model       = config['parameters_mnist']['current_model']
        noising             = config['opacus_mnist']['noising']
        clipping            = config['opacus_mnist']['clipping']
        delta               = config['opacus_mnist']['delta']
        target_epsilon      = config['opacus_mnist']['target_epsilon']
    else:
        current_dataset = 'WBC'
        num_epochs          = config['parameters_wbc']['num_epochs']
        batch_size          = config['parameters_wbc']['batch_size']
        learning_rate       = config['parameters_wbc']['learning_rate']
        general_seed        = config['parameters_wbc']['general_seed']
        num_trainings       = config['parameters_wbc']['num_trainings']
        length_dataset      = config['parameters_wbc']['length_dataset']
        outliers_indices    = config['parameters_wbc']['outliers_indices']
        current_model       = config['parameters_wbc']['current_model']
        noising             = config['opacus_wbc']['noising']
        clipping            = config['opacus_wbc']['clipping']
        delta               = config['opacus_wbc']['delta']
        target_epsilon      = config['opacus_wbc']['target_epsilon']

    return (current_dataset, num_epochs, batch_size, learning_rate, 
            general_seed, num_trainings, length_dataset, outliers_indices, 
            current_model, noising, clipping, delta, target_epsilon)

def get_exp_params(exp_config):
    make_private        = exp_config['settings']['make_private']
    data_processing     = exp_config['settings']['data_processing']
    dataset_mode        = exp_config['settings']['dataset_mode']
    init_mode           = exp_config['settings']['init_mode']
    dp_mode             = exp_config['settings']['dp_mode']

    return (make_private, data_processing, dataset_mode, init_mode, dp_mode)
