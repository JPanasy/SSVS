import numpy as np
import os
import ntpath
import time
from . import util, html

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these images
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """Visualizer that logs losses to TensorBoard and saves images to HTML.

    This version removes the Visdom dependency and uses TensorBoard instead.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class."""
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False

        self.tb_writer = None
        if SummaryWriter is not None:
            tb_dir = os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard')
            util.mkdirs(tb_dir)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
            print('TensorBoard logging to %s' % tb_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status."""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """Save current results to HTML and TensorBoard."""
        if self.tb_writer is not None:
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                image_chw = image_numpy.transpose([2, 0, 1])
                self.tb_writer.add_image(label, image_chw, global_step=epoch)

        if self.use_html and (save_result or not self.saved):
            self.saved = True
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """Log current losses to TensorBoard."""
        step = epoch + counter_ratio
        if self.tb_writer is not None:
            for k, v in losses.items():
                self.tb_writer.add_scalar(k, float(v), global_step=step)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Print current losses on console and save them to disk."""
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def close(self):
        """Close any open loggers."""
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
