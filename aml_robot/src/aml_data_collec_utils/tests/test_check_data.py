import numpy as np
import quaternion

import aml_robot

from aml_data_collec_utils.record_sample import Sample


sample = Sample()

data = sample.get_sample(sample_id=1)


print data['sample_id']

print len(data['state'])

# print data['state']

print len(data['task_action']) 

print len(data['task_effect'])

print data['task_effect'][2]

print data['task_status'] 