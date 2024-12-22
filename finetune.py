import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, get_trainer, set_color

from unisrec import UniSRec
from data.dataset import UniSRecDataset


def finetune(dataset, pretrained_file, fix_enc=True, **kwargs):
    """
    Fine-tunes the UniSRec model on the specified dataset with optional pre-trained weights.

    Args:
        dataset (str): Name of the dataset to use.
        pretrained_file (str): Path to the pre-trained model file.
        fix_enc (bool): Whether to fix the encoder parameters during fine-tuning.
        **kwargs: Additional keyword arguments for configuration.

    Returns:
        tuple: Contains the model, dataset name, and a dictionary of results.
    """
    # Configuration files initialization
    props = ['props/UniSRec.yaml', 'props/finetune.yaml']
    print(f"Configuration files: {props}")

    # Initialize configuration with additional keyword arguments
    config = Config(
        model=UniSRec,
        dataset=dataset,
        config_file_list=props,
        config_dict=kwargs
    )

    # Initialize random seed for reproducibility
    init_seed(config['seed'], config['reproducibility'])

    print("+" * 50)
    config['valid_neg_sample_args']['sample_num'] = 29
    config['valid_neg_sample_args']['distribution'] = 'popularity'
    config['test_neg_sample_args']['sample_num'] = 29
    config['test_neg_sample_args']['distribution'] = 'popularity'
    config['train_neg_sample_args']['distribution'] = 'popularity'
    config['train_neg_sample_args']['sample_num'] = 29
    config['train_neg_sample_args']['alpha'] = 1
    config['eval_args']['mode']['valid'] = 'pop29'
    config['eval_args']['mode']['test'] = 'pop29'
    print("+" * 50)
    init_logger(config)
    logger = getLogger()
    logger.info("Configuration:")
    logger.info(config)

    # Log the validation and test negative sampling arguments
    logger.info(f"Validation Negative Sampling Args: {config['valid_neg_sample_args']}")
    logger.info(f"Test Negative Sampling Args: {config['test_neg_sample_args']}")

    # Dataset filtering and preparation
    dataset = UniSRecDataset(config)
    logger.info(f"Dataset Summary: {dataset}")

    # Split the dataset into training, validation, and test sets
    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info("Data Preparation Completed.")

    # Initialize the model and move it to the specified device (CPU/GPU)
    model = UniSRec(config, train_data.dataset).to(config['device'])
    logger.info(f"Model Initialized: {model}")

    # Load pre-trained model weights if provided
    if pretrained_file:
        checkpoint = torch.load(pretrained_file, map_location=config['device'])
        logger.info(f'Loading pre-trained model from: {pretrained_file}')
        logger.info(f'Transferring knowledge from dataset [{checkpoint["config"]["dataset"]}] to [{dataset}]')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        if fix_enc:
            logger.info("Fixing encoder parameters.")
            for param in model.position_embedding.parameters():
                param.requires_grad = False
            for param in model.trm_encoder.parameters():
                param.requires_grad = False

    # Initialize the trainer based on the model type
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    logger.info("Trainer Initialized.")

    # Start the fine-tuning (training) process
    logger.info("Starting Training Process...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config['show_progress']
    )
    logger.info("Training Completed.")

    # Evaluate the model on the test set using the best validation model
    logger.info("Starting Evaluation on Test Set...")
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config['show_progress']
    )
    logger.info("Evaluation Completed.")

    # Log the best validation and test results
    logger.info(set_color('Best Validation Result:', 'yellow') + f' {best_valid_result}')
    logger.info(set_color('Test Result:', 'yellow') + f' {test_result}')

    return config['model'], config['dataset'], {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


if __name__ == '__main__':
    """
    Entry point for the fine-tuning script. Parses command-line arguments and initiates the fine-tuning process.
    """
    parser = argparse.ArgumentParser(description="Fine-tune the UniSRec model with specified configurations.")

    # Command-line arguments
    parser.add_argument(
        '-d',
        type=str,
        default='Scientific',
        help='Name of the dataset to use (default: Scientific)'
    )
    parser.add_argument(
        '-p',
        type=str,
        default='',
        help='Path to the pre-trained model file (default: empty string)'
    )
    parser.add_argument(
        '-f',
        type=bool,
        default=True,
        help='Whether to fix encoder parameters during fine-tuning (default: True)'
    )
    args, unparsed = parser.parse_known_args()
    print(f"Parsed Arguments: {args}")

    # Since we're using YAML configuration, no need to define negative sampling here
    # Just call the finetune function with the parsed arguments
    finetune(
        dataset=args.d,
        pretrained_file=args.p,
        fix_enc=args.f
    )
