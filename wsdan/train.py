import tensorflow as tf
import yaml
import argparse
import os
import time

from tf_2.finegrained.wsdan.utils.image_utils import visualize
from tf_2.finegrained.wsdan.utils.image_utils import attention_crop, attention_drop
from tf_2.finegrained.wsdan.dataset.datahandler import PangyoDataHandler, CubDataHandler
from tf_2.finegrained.wsdan.models.wsdan import WSDAN
from tf_2.finegrained.wsdan.losses import WSDANLoss


class WSDANTrainer:
    def __init__(self, configure):
        self.cfg = configure

        self.crop_threshold = self.cfg['crop_threshold']
        self.drop_threshold = self.cfg['drop_threshold']
        self.object_threshold = self.cfg['object_threshold']
        self.crop_heat_map_threshold = self.cfg['crop_heat_map_threshold']
        self.drop_heat_map_threshold = self.cfg['drop_heat_map_threshold']
        self.beta = self.cfg['beta']

        self.num_class = self.cfg['num_class']
        self.num_attentions = self.cfg['num_attentions']
        self.initial_learning_rate = self.cfg['initial_learning_rate']
        self.lr_decay_steps = self.cfg['lr_decay_steps']
        self.lr_decay_rate = self.cfg['lr_decay_rate']
        self.initial_weight_decay = self.cfg['initial_weight_decay']
        self.wd_decay_steps = self.cfg['wd_decay_steps']
        self.wd_decay_rate = self.cfg['wd_decay_rate']
        self.momentum = self.cfg['momentum']
        self.num_epochs = self.cfg['num_epochs']
        self.num_batch = self.cfg['num_batch']
        self.image_fmt = self.cfg['image_fmt']
        self.is_shuffle = self.cfg['is_shuffle']
        self.pre_trained_type = self.cfg['pre_trained_type']
        self.resize_h_w = self.cfg['resize_h_w']

        self.dataset_name = self.cfg['dataset_name']
        self.image_dir = self.cfg['image_dir']
        self.train_dir = self.cfg['train_dir']
        self.valid_dir = self.cfg['valid_dir']
        self.ckpt_dir = self.cfg['ckpt_dir']
        self.logs_dir = self.cfg['logs_dir']

        self.train_log_fmt = self.cfg['train_log_fmt'].format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        self.valid_log_fmt = self.cfg['valid_log_fmt'].format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))

        self.ckpt_path = self.cfg['ckpt_path']

        self.train_writer_path = os.path.join(self.logs_dir, self.train_log_fmt)
        self.valid_writer_path = os.path.join(self.logs_dir, self.valid_log_fmt)

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.train_writer_path, exist_ok=True)
        os.makedirs(self.valid_writer_path, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.train_paths, self.valid_paths = self.create_data_paths()
        self.train_dataset, self.valid_dataset = self.create_batch_dataset()
        self.model = self.build_model()
        self.loss_fn = self.create_loss_fn(self.num_class)
        self.lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_learning_rate, self.lr_decay_steps, self.lr_decay_rate, staircase=False, name=None)
        self.wd_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            self.initial_weight_decay, self.wd_decay_steps, self.wd_decay_rate, staircase=False, name=None)
        self.optimizer = self.create_optimizer()

        self.train_summary_writer = tf.summary.create_file_writer(self.train_writer_path)
        self.val_summary_writer = tf.summary.create_file_writer(self.valid_writer_path)

    def create_step_fn(self, mode='train'):
        def step_fn(images, labels):
            with tf.GradientTape() as tape:
                logits_1, attention_maps, embeddings = self.model(images)
                crop_images, drop_images, crop_attentions, drop_attentions = self.get_augmented_images(images,
                                                                                                       attention_maps)
                logits_2, _, _ = self.model(crop_images)
                logits_3, _, _ = self.model(drop_images)
                total_loss, classification_loss, attention_loss = self.loss_fn(labels,
                                                                               logits_1,
                                                                               logits_2,
                                                                               logits_3,
                                                                               embeddings,
                                                                               self.beta)

            if mode == 'train':
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                return total_loss, classification_loss, attention_loss,  crop_images, drop_images, crop_attentions, drop_attentions
            elif mode == 'valid':

                return total_loss, classification_loss, attention_loss, crop_images, drop_images, crop_attentions, drop_attentions
            else:

                raise print('Please Check mode : {}'.format(mode))

        return step_fn

    def train_step(self, images, labels):

        return self.create_step_fn(mode='train')(images, labels)

    def valid_step(self, images, labels):

        return self.create_step_fn(mode='valid')(images, labels)

    def get_augmented_images(self, images, attention_maps):
        attention_maps = tf.image.resize(attention_maps, [images.shape[1], images.shape[2]])

        # attention crop
        bboxes, crop_attentions = tf.map_fn(attention_crop, attention_maps, dtype=(tf.float32, tf.float32))
        box_ind = tf.range(self.num_batch, dtype=tf.int32)
        crop_images = tf.image.crop_and_resize(images, bboxes, box_ind,
                                               crop_size=[images.shape[1], images.shape[2]])

        # attention drop
        masks, drop_attentions = tf.map_fn(attention_drop, attention_maps, dtype=(tf.float32, tf.float32))
        drop_images = images * masks

        return crop_images, drop_images, crop_attentions, drop_attentions

    def build_model(self):
        print('build model ...')
        try:
            model = WSDAN(self.num_class, self.num_attentions)

            if self.pre_trained_type == '':
                pass

            elif self.pre_trained_type == 'latest' and os.path.exists(self.ckpt_dir):
                latest = tf.train.latest_checkpoint(self.ckpt_dir)
                model.load_weights(latest)

            elif self.pre_trained_type == 'specified' and os.path.exists(self.ckpt_path):
                model.load_weights(self.ckpt_path)

            else:
                raise print('Please Check pre_trained_type : {}'.format(self.pre_trained_type))

            return model
        except Exception as e:

            raise print(e)

    def create_data_paths(self):
        if self.dataset_name == 'pangyo':
            train_paths = [os.path.join(self.train_dir, file_name)
                           for file_name in os.listdir(self.train_dir)
                           if file_name.endswith(self.image_fmt)]
            valid_paths = [os.path.join(self.valid_dir, file_name)
                           for file_name in os.listdir(self.valid_dir)
                           if file_name.endswith(self.image_fmt)]

        elif self.dataset_name == 'cub':
            print('create_cub_batch_dataset ... ')
            # TODO : train, valid, test 데이터를 한 폴더에 몰아넣지 말고, 폴더 별로 구분해서 사용할 수 있도록 작업하기.
            file_folders = [os.path.join(self.image_dir, folder) for folder in os.listdir(self.image_dir) if not folder.startswith('.')]

            train_paths = sum([list(map(lambda x: os.path.join(file_folder, x),
                                        sorted(os.listdir(file_folder))[:int(len(os.listdir(file_folder))*0.75)]))
                               for file_folder in file_folders], [])
            valid_paths = sum([list(map(lambda x: os.path.join(file_folder, x),
                                        sorted(os.listdir(file_folder))[int(len(os.listdir(file_folder))*0.75):int(len(os.listdir(file_folder))*0.9)]))
                               for file_folder in file_folders], [])

        else:
            print('Please check dataset name is correct!! : {}'.format(self.dataset_name))
            print('you can select in [pangyo, cub]')

            return None, None

        return train_paths, valid_paths

    def create_batch_dataset(self):
        print('create_batch_dataset ... ')
        if self.dataset_name == 'pangyo':
            data_handler = PangyoDataHandler(self.resize_h_w)
        elif self.dataset_name == 'cub':
            data_handler = CubDataHandler(self.resize_h_w)
        else:

            return None, None

        train_dataset = data_handler.create_batch_dataset(self.train_paths, self.num_batch,
                                                          shuffle=self.is_shuffle)
        valid_dataset = data_handler.create_batch_dataset(self.valid_paths, self.num_batch)
        print('train dataset : {}, valid_dataset : {}'.format(len(self.train_paths), len(self.valid_paths)))

        return train_dataset, valid_dataset

    def create_optimizer(self):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.lr_scheduler, momentum=self.momentum)

        return optimizer

    @staticmethod
    def create_loss_fn(num_class):
        print('create loss fn ...')
        loss_fn = WSDANLoss(num_class).calculate_total_loss

        return loss_fn

    def train(self):
        print('train start !')

        avg_train_total_loss = 0.
        avg_train_classification_loss = 0.
        avg_train_attention_loss = 0.
        avg_valid_total_loss = 0.
        avg_valid_classification_loss = 0.
        avg_valid_attention_loss = 0.
        heat_map = None

        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_dataset):
                total_loss, classification_loss, attention_loss, crop_images, \
                drop_images, crop_attentions, drop_attentions = self.train_step(images, labels)

                heat_map = visualize(images[0, ...], crop_attentions[0, ...], drop_attentions[0, ...],
                                     crop_images[0, ...], drop_images[0, ...])
                heat_map = tf.gather(heat_map, [2, 1, 0], axis=-1)
                heat_map = tf.expand_dims(heat_map, axis=0)

                avg_train_total_loss = (avg_train_total_loss * i + total_loss) / (i+1)
                avg_train_classification_loss = (avg_train_classification_loss * i + classification_loss) / (i+1)
                avg_train_attention_loss = (avg_train_attention_loss * i + attention_loss) / (i+1)

                if i % 10 == 0:
                    print("epoch : {} | batch: {} | avg_train_total_loss : {} | avg_train_classification_loss : {} | avg_train_attention_loss : {}"
                          .format(epoch, i, avg_train_total_loss, avg_train_classification_loss, avg_train_attention_loss))

            with self.train_summary_writer.as_default():
                tf.summary.scalar('avg_total_loss', avg_train_total_loss, step=epoch)
                tf.summary.scalar('avg_classification_loss', avg_train_total_loss, step=epoch)
                tf.summary.scalar('avg_attention_loss', avg_train_attention_loss, step=epoch)
                tf.summary.image('heat_map_image', heat_map / 255., step=epoch)

            if epoch % 10 == 0:
                saved_model_path = self.ckpt_dir + "epoch{}_{}". \
                    format(epoch, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                self.model.save_weights(saved_model_path)

            for i, (images, labels) in enumerate(self.valid_dataset):
                total_loss, classification_loss, attention_loss, crop_images, \
                drop_images, crop_attentions, drop_attentions = self.valid_step(images, labels)

                avg_valid_total_loss = (avg_valid_total_loss * i + total_loss) / (i+1)
                avg_valid_classification_loss = (avg_valid_classification_loss * i + classification_loss) / (i+1)
                avg_valid_attention_loss = (avg_valid_attention_loss * i + attention_loss) / (i+1)

                if i % 10 == 0:
                    print("epoch : {} | batch : {} | avg_valid_total_loss : {} | avg_valid_classification_loss : {} | avg_valid_attention_loss : {}"
                          .format(epoch, i, avg_valid_total_loss, avg_valid_classification_loss, avg_valid_attention_loss))

            with self.val_summary_writer.as_default():
                tf.summary.scalar('avg_total_loss', avg_valid_total_loss, step=epoch)
                tf.summary.scalar('avg_classification_loss', avg_valid_classification_loss, step=epoch)
                tf.summary.scalar('avg_attention_loss', avg_valid_attention_loss, step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-path', default='./train.yml')
    args = parser.parse_args()

    with open(args.config_file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    trainer = WSDANTrainer(config)
    trainer.train()
