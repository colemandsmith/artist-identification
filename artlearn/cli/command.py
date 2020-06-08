import click
import torchvision.models.resnet as resnet

from artlearn.common_utils import (
    LOG_DIR, MODEL_DIR,
    get_dataloaders, ArtistLearner
)


@click.command()
@click.option('--mode', type=str, default='sgd',
              help='Which optimizer you wish to use, currently supports '
                   'SGD and ADAM. Default is SGD.')
@click.option('-e', '--epochs', type=int, default=80,
              help='Number of epochs with which to train. Default is 80.')
@click.option('-l', '--lr', type=float, default=1e-3,
              help='The learning rate to use for the optimizer. '
                   'Default is 1e-3.')
@click.option('-m', '--momentum', type=float, default=0.9,
              help='If using SGD, the momentum to use. Default is 0.9.')
@click.option('-a', '--log-after', type=int, default=80,
              help='Number of iterations within an epoch to log out stats '
                   'after. Default is 80.')
@click.option('--log-path', envvar='ART_LOG_PATH', type=str, default=LOG_DIR,
              help='Absolute path to write logs out to.')
@click.option('--model-path', envvar='ART_MODEL_PATH', type=str,
              default=MODEL_DIR,
              help='Absolute path to write model files out to.')
@click.option('-n', '--name', type=str,
              help='Name override for the model and log files. Otherwise, '
                   'named after its parameters in the form: '
                   '{mode}_e_{epochs}_lr_{lr}_m_{momentum}')
@click.option('-p', '--pretrained', is_flag=True)
def train(mode, epochs, lr, momentum, log_after, log_path, model_path, name,
          pretrained):
    train, test, val = get_dataloaders()
    network = resnet.resnet18(pretrained=pretrained)
    learner = ArtistLearner(network, mode, epochs, train, test, val, lr=lr,
                            momentum=momentum, log_after=log_after,
                            log_path=log_path, model_path=model_path,
                            model_name=name)
    learner.train_and_validate()


def main():
    train()


if __name__ == '__main__':
    main()
