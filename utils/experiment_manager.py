
import tensorflow as tf
import fnmatch
import os

def make_checkpoint_path(base_path, FLAGS):
  # make checkpoint path with all the flags specifing different directories

  # run through all params and add them to the base path
  for k, v in FLAGS.__dict__['__flags'].items():
    if k != 'base_dir':
      base_path = base_path + '/' + k + '_' + str(v)

  return base_path

def list_all_checkpoints(base_path, FLAGS):
  # get a list off all the checkpoint directorys

  # run through all params and add them to the base path
  paths = []
  for root, dirnames, filenames in os.walk(base_path):
    for filename in fnmatch.filter(filenames, 'checkpoint'):
      paths.append(root)
  print(paths)
  return paths

def set_flags_given_checkpoint_path(path, FLAGS):
  # get a list off all the checkpoint directorys

  # run through all params and add them to the base path
  split_path = path.split('/')
  for param in split_path:
    [param_name, param_value] = param.split('_')
    param_type = type(FLAGS.__dict__['__flags'][param_name])
    FLAGS.__dict__['__flags'][param_name] = param_type(param_value)




