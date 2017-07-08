
import sys
sys.path.append('../')
import utils.que as que

q = que.Que([0])
q.enque_file("flow_experiments.txt")
#q.enque_file("boundary_experiments.txt")
q.start_que_runner()





