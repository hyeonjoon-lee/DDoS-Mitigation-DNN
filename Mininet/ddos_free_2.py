import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import pickle


def generate_scalar():
    num_samples = 50000

    attack = pd.read_csv('/home/ubuntu/pox/pox/forwarding/dataset_attack.csv', nrows=num_samples)
    normal = pd.read_csv('/home/ubuntu/pox/pox/forwarding/dataset_normal.csv', nrows=num_samples)

    normal.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'class']
    attack.columns = ['frame.len', 'frame.protocols', 'ip.hdr_len', 'ip.len', 'ip.flags.rb', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'ip.src', 'ip.dst', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size', 'tcp.time_delta', 'class']

    normal = normal.drop(['frame.protocols', 'ip.flags.rb', 'ip.src', 'ip.dst', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.time_delta'], axis=1)
    attack = attack.drop(['frame.protocols', 'ip.flags.rb', 'ip.src', 'ip.dst', 'tcp.flags.res', 'tcp.flags.ns', 'tcp.flags.cwr', 'tcp.flags.ecn', 'tcp.flags.urg', 'tcp.time_delta'], axis=1)

    features = ['frame.len', 'ip.hdr_len', 'ip.len', 'ip.flags.df', 'p.flags.mf', 'ip.frag_offset', 'ip.ttl', 'ip.proto', 'tcp.srcport', 'tcp.dstport', 'tcp.len', 'tcp.ack', 'tcp.flags.ack', 'tcp.flags.push', 'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.fin', 'tcp.window_size']

    normal = normal[features].values
    attack = attack[features].values
    data = np.concatenate((normal, attack))

    scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
    scalar.fit(data)

    pickle.dump(scalar, open('myScalar2.pkl', 'wb'))
    print "dump finished"


if __name__ == "__main__":
    generate_scalar()
