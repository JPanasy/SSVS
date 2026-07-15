import numpy as np
import os
import ntpath
import time
from . import util, html

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

try:
    import visdom
except Exception:
    visdom = None

try:
    import wandb
except Exception:
    wandb = None


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
    """Visualizer that logs to Visdom/TensorBoard and saves images to HTML."""

    def __init__(self, opt):
        """Initialize the Visualizer class."""
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.saved = False
        self.display_single_pane_ncols = opt.display_ncols

        self.use_visdom = self.display_id > 0 and visdom is not None
        self.vis = None
        self.plot_data = None
        if self.use_visdom:
            self.vis = visdom.Visdom(
                server=opt.display_server,
                port=opt.display_port,
                env=opt.display_env,
                use_incoming_socket=False,
            )
            try:
                visdom_connected = self.vis.check_connection()
            except Exception as exc:
                print('Visdom connection failed at %s:%s (env=%s): %s' % (
                    opt.display_server,
                    opt.display_port,
                    opt.display_env,
                    exc,
                ))
                visdom_connected = False
            if not visdom_connected:
                print('Visdom server not reachable at %s:%s (env=%s).' % (opt.display_server, opt.display_port, opt.display_env))
                print('Start with: python -m visdom.server -p %s' % opt.display_port)
                self.use_visdom = False

        self.wandb_run = None
        self.use_wandb = bool(getattr(opt, 'use_wandb', False)) and getattr(opt, 'wandb_mode', 'offline') != 'disabled'
        if self.use_wandb:
            if wandb is None:
                print('W&B logging requested but wandb is not installed. Install with: pip install wandb')
                self.use_wandb = False
            else:
                wandb_dir = os.path.join(opt.checkpoints_dir, opt.name, 'wandb')
                util.mkdirs(wandb_dir)
                init_kwargs = dict(
                    project=getattr(opt, 'wandb_project', 'SSVS'),
                    name=opt.name,
                    mode=getattr(opt, 'wandb_mode', 'offline'),
                    dir=wandb_dir,
                    config=vars(opt),
                    reinit=True,
                )
                if getattr(opt, 'wandb_entity', ''):
                    init_kwargs['entity'] = opt.wandb_entity
                self.wandb_run = wandb.init(**init_kwargs)
                print('W&B logging enabled in %s mode at %s' % (getattr(opt, 'wandb_mode', 'offline'), wandb_dir))

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

    def display_current_results(self, visuals, epoch, save_result, total_iters=None):
        """Display/save current results to Visdom, TensorBoard and HTML."""
        if self.use_visdom:
            if self.display_single_pane_ncols > 0:
                ncols = self.display_single_pane_ncols
                images = []
                labels = []
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    images.append(image_numpy.transpose([2, 0, 1]))
                    labels.append(label)

                while len(images) % ncols != 0:
                    images.append(np.zeros_like(images[0]))
                    labels.append('')

                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                opts=dict(title=self.name + ' images'))
                label_html = '<table><tr>{}</tr></table>'.format(
                    ''.join(['<td>{}</td>'.format(l) for l in labels]))
                self.vis.text(label_html, win=self.display_id + 2,
                              opts=dict(title=self.name + ' labels'))
            else:
                for i, (label, image) in enumerate(visuals.items()):
                    image_numpy = util.tensor2im(image)
                    self.vis.image(image_numpy.transpose([2, 0, 1]),
                                   opts=dict(title=label),
                                   win=self.display_id + i)

        if self.tb_writer is not None:
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                image_chw = image_numpy.transpose([2, 0, 1])
                self.tb_writer.add_image(label, image_chw, global_step=epoch)

        if self.wandb_run is not None:
            step = int(total_iters) if total_iters is not None else int(epoch)
            wandb_images = {}
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                wandb_images['images/%s' % label] = wandb.Image(image_numpy, caption='epoch %d' % epoch)
            if wandb_images:
                self.wandb_run.log(wandb_images, step=step)

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

    def plot_current_losses(self, epoch, counter_ratio, losses, total_iters=None):
        """Log current losses to Visdom and TensorBoard."""
        step = epoch + counter_ratio
        if self.use_visdom:
            if self.plot_data is None:
                self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
            self.plot_data['X'].append(step)
            self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
            x = np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
            y = np.array(self.plot_data['Y'])
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'
                },
                win=self.display_id,
            )

        if self.tb_writer is not None:
            for k, v in losses.items():
                self.tb_writer.add_scalar(k, float(v), global_step=step)

        if self.wandb_run is not None:
            wandb_step = int(total_iters) if total_iters is not None else int(step * 1000)
            self.wandb_run.log({'loss/%s' % k: float(v) for k, v in losses.items()}, step=wandb_step)

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
        if self.wandb_run is not None:
            self.wandb_run.finish()
