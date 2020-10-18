from abc import ABC
from jsonrpcclient.clients.http_client import HTTPClient
import pickle
from time import sleep


class RemoteCommunicator(ABC):
    """Send data to a remote executor."""

    def __init__(self):
        pass

    def submit_job(self, *args, **kwargs):
        raise NotImplementedError

    def get_result(self, *args, **kwargs):
        raise NotImplementedError


class RemoteHTTPPickleCommunicator(RemoteCommunicator):
    """Send requests via TCP/pickle files."""

    def __init__(self, http_remote_port):
        super().__init__()
        self.client = HTTPClient(http_remote_port)

    def submit_job(self, client_id, data, data_path, answer_path):
        # saving pickle data
        pickle.dump(data, open(data_path, 'wb'))

        # connecting to the RPC server
        result = self.client.process(client_id, uid=0, data_path=data_path, answer_path=answer_path).data.result

        assert result is True, str(result)

    def get_result(self, answer_path):
        # loading weights and information
        # busy wait with a delay
        while True:
            try:
                answer = pickle.load(open(answer_path, 'rb'))
                break
            except Exception as e:
                print(e, "Waiting")
                sleep(0.5)

        return answer
