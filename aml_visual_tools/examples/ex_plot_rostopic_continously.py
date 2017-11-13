from baxter_core_msgs.msg import EndpointState
from aml_visual_tools.plot_data_stream import VisualizeROStopic

def main():
    config = {}
    config['plot_name']  = 'Wrench Force'
    config['rostopic']   = '/robot/limb/right/endpoint_state'
    config['msg_type']   =  EndpointState
    #following msg_fields denote how deep you have to go
    #for example goes rosotpic_data.wrench.force
    config['msg_fields'] = ['wrench', 'force'] 
    config['figsize']    = (10,10)

    viz_data = VisualizeROStopic(config)
    viz_data.run()

if __name__ == '__main__':
    main()