
import os
import process
import time

class Que:
  def __init__(self, avalible_gpus=[0,1]):
    self.pl = []
    self.running_pl = []
    self.avalible_gpus=avalible_gpus

  def enque_file(self, file_name):
    cmd_list = [line.rstrip('\n') for line in open(file_name)]
    for cmd in cmd_list:
      broken_up_cmd = cmd.split()
      self.pl.append(process.Process(broken_up_cmd))

  def start_next(self, gpu):
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Not Started":
        self.pl[i].start(gpu)
        break

  def find_free_gpu(self):
    used_gpus = []
    for i in xrange(len(self.pl)):
      if self.pl[i].get_status() == "Running":
        used_gpus.append(self.pl[i].get_gpu())
    free_gpus = list(set(self.avalible_gpus) - set(used_gpus)) 
    return free_gpus

  def update_pl_status(self):
    for i in xrange(len(self.pl)):
      self.pl[i].update_status()

  def print_que_status(self):
    print(chr(27) + "[2J")
    print("QUE STATUS")
    for i in xrange(len(self.pl)):
      self.pl[i].print_info()
 
  def start_que_runner(self):
    while True:
      time.sleep(1)
      free_gpus = self.find_free_gpu()
      for gpu in free_gpus:
        self.start_next(gpu)
      self.update_pl_status()
      self.print_que_status()
      


