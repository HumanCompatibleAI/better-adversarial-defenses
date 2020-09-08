#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from stable_baselines_external_data import SBPPORemoteData
import pickle
import numpy as np
from matplotlib import pyplot as plt

import socket
import sys
from multiprocessing import Queue as queue
import pickle
import multiprocessing
import traceback


# In[ ]:


class MultipleWorker(object):
    """Handles multiple workers identified by IDs."""
    
    def __init__(self):
        """Initialize with zero workers."""
        self.workers = set()
        self.kwargs_queues = {}
        self.return_queues = {}
        self.worker_processes = {}
        assert hasattr(self, '_process_fcn'), "Please implement _process_fcn"
        self._process_init()
        
    def target_process(self, worker_id):
        """Process to run in each worker."""
        #print("Starting process", worker_id)
        while True:
            kwargs = pickle.loads(self.kwargs_queues[worker_id].get())
            try:
                result = self._process_fcn(**kwargs, worker_id=worker_id)
            except Exception as e:
                result = ("Exception", traceback.format_exc())
                print(result[1])
            self.return_queues[worker_id].put(pickle.dumps(result))
            
    def _process_init(self):
        """Initialize client code."""
        pass
            
    def _process_fcn(self, **kwargs):
        """Process function, to be implemented."""
        #print("Called with:", kwargs)
        return str(kwargs)
        
    def new_worker(self, worker_id):
        """Create a worker with a given ID."""
        if worker_id in self.workers:
            return
        else:
            self.workers.add(worker_id)
            self.kwargs_queues[worker_id] = queue()
            self.return_queues[worker_id] = queue()
            self.worker_processes[worker_id] = multiprocessing.Process(target=self.target_process,
                                                                       kwargs=dict(worker_id=worker_id))
            self.worker_processes[worker_id].start()
            
    def process(self, worker_id, kwargs):
        """Process a request."""
        if worker_id not in self.workers:
            print("Creating worker", worker_id)
            self.new_worker(worker_id)
        self.kwargs_queues[worker_id].put(pickle.dumps(kwargs))
        res = self.return_queues[worker_id].get()
        return pickle.loads(res)
    
    def __del__(self):
        """Close all processes."""
        for p in self.worker_processes.values():
            p.kill()
            
class MultiStepTrainer(object):
    """Train with stable baselines on external data, supporting multiple trainers."""
    def __init__(self):
        self.trainers = {}
    def create(self, uid, config):
        if uid in self.trainers:
            print("Trainer %s already exists, doing nothing" % uid)
        else:
            self.trainers[uid] = SBPPORemoteData(config=config)
    def process(self, uid, rollouts, weights):
        if uid not in self.trainers:
            print("Error: trainer %s does not exist" % uid)
            return None
        
        self.trainers[uid].set_weights(weights)
        info = self.trainers[uid].learn(rollouts)
        new_weights = self.trainers[uid].get_weights()
        return {'info': info, 'weights': new_weights}
            
class MultipleWorkerTrainer(MultipleWorker):
    """Train with multiple workers."""
    
    def _process_init(self):
        self.trainer = None
    
    def _process_fcn(self, uid, config, data_path, answer_path, worker_id):
        if self.trainer is None:
            self.trainer = MultiStepTrainer()
        print("Process call", uid, data_path, worker_id)
        self.trainer.create(uid, config)
        data = pickle.load(open(data_path, 'rb'))
        rollouts = data['rollouts']
        weights = data['weights']
        
        result = self.trainer.process(uid, rollouts, weights)
        
        pickle.dump(result, open(answer_path, 'wb'))
        return True


# In[5]:


from asgiref.sync import async_to_sync
from tornado import ioloop, web
from jsonrpcserver import method, dispatch as dispatch, serve
import argparse

parser = argparse.ArgumentParser(description='Launch the multiprocess stable baselines server.')
parser.add_argument('--port', metavar='N', default=50001,
                    help='port to listen on')


# Server for DatabasePreferenceLearner
class MainHandler(web.RequestHandler):
    def post(self):
        request = self.request.body.decode()
        print(request)
        response = dispatch(request)
        print(response)
        if response.wanted:
            self.write(str(response))

app = web.Application([(r"/", MainHandler)])
trainer = None
            
def run_server(port=50001):
    """Run server."""

    print("Listening on port %d" % port)

    global trainer
    trainer = MultipleWorkerTrainer()

    @method
    def process(*args, **kwargs):
        global trainer
        return trainer.process(*args, **kwargs)
    serve(port=port)

if __name__ == "__main__":
    args = parser.parse_args()
    run_server(port=args.port)


