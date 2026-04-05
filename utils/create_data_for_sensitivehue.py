
import pickle as pk
import os
import numpy as np

data_name = 'SMD'
root_path = '../data/SMD/'
data_path = 'SMD.pk'
num_entities = 28

#######################
output_path = '../data/output/'
os.makedirs(output_path, exist_ok = True)

pk_path = os.path.join(root_path, data_path)
with open(pk_path, 'rb') as file:
    data = pk.load(file)

# here the entity idx: from  1 to n

train_len = 0
test_len = 0
num_anom = 0
for entity_id in range(1, num_entities + 1):
    data_normal = data['x_trn'][entity_id - 1].astype(np.float32) # entity id starts from 1, however index strt from 0
    data_attack = data['x_tst'][entity_id - 1].astype(np.float32) # entity id starts from 1, however index strt from 0
    attack_label = data['lab_tst'][entity_id - 1].astype(np.int32)
    
    train_len += data_normal.shape[0]
    test_len += data_attack.shape[0]
    num_anom += sum(attack_label)
    
    entity_path = os.path.join(output_path, '{}_{}'.format(data_name, entity_id))
    os.makedirs(entity_path, exist_ok = True)
    test = {'x': data_attack, 'y': attack_label}
    
    # np.save(os.path.join(entity_path, 'train.npy'), data_normal)
    # np.savez(os.path.join(entity_path, 'test.npz'), **test)
    
print(f'Total Train: {train_len}')
print(f'Total Test: {test_len}')
print(f'anomaly ratio: {num_anom / test_len}')
    
    
    
    
    