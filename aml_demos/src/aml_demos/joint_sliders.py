#!/usr/bin/env python
'''
A small app for controlling Baxter's joint positions using sliders

usage: ./joint_sliders.py (left|right)

author: Matthew Broadway (https://github.com/mbway)
'''

import os
import sys
import time
import threading
import signal
import numpy as np

# use the pyqt4 abstraction provided by ROS
import python_qt_binding.QtCore as qtc
import python_qt_binding.QtGui as qtg

# ROS imports
import rospy
from aml_robot.baxter_robot import BaxterArm


class FloatSlider(qtg.QSlider):
    ''' QSliders cannot have floating point values, only integers! '''
    def __init__(self, *args, **kwargs):
        super(FloatSlider, self).__init__(*args, **kwargs)
        self._float_min, self._float_max = 0.0, 0.0
        self.set_steps(1000)
    def set_steps(self, steps):
        self._steps = steps # the underlying integer slider varies from 0 to self._steps
        self.setRange(0, self._steps)
    def set_float_range(self, lower, upper):
        self._float_min, self._float_max = float(lower), float(upper)
    def get_float_value(self):
        float_range = self._float_max - self._float_min
        return self._float_min + (self.value()/float(self._steps) * float_range)
    def set_float_value(self, val):
        float_range = self._float_max - self._float_min
        val = (val - self._float_min)/float_range * self._steps
        super(FloatSlider, self).setValue(val)

def test_float_slider():
    app = qtg.QApplication(sys.argv)
    fs = FloatSlider()
    fs.set_float_range(0, 1000)
    fs.set_float_value(50)
    assert abs(fs.get_float_value()-50) < 1e-5
    fs.set_steps(100000)
    fs.set_float_range(-223.9, 100.5)
    fs.set_float_value(-150.45)
    assert abs(fs.get_float_value()-(-150.45)) < 1e-2
    print('float slider tests passed')
    sys.exit(0)
#test_float_slider()

# easier to see what the joint does than the given names
my_descriptions = {
    's0' : '(1-twist)',
    's1' : '(2-swing)', 'e0' : '(2-twist)',
    'e1' : '(3-swing)', 'w0' : '(3-twist)',
    'w1' : '(4-swing)', 'w2' : '(4-twist)',
}

class StatePopup(qtg.QWidget):
    def __init__(self, parent):
        super(StatePopup, self).__init__(parent)
        self.parent = parent
        self.ignore = {'inertia', 'jacobian', 'gravity_comp', 'depth_image', 'rgb_image'}
        self.setWindowFlags(qtc.Qt.Window)

        vbox = qtg.QVBoxLayout()

        refresh = qtg.QPushButton('refresh')
        refresh.clicked.connect(self.refresh)
        vbox.addWidget(refresh)

        self.textbox = qtg.QTextEdit()
        font = qtg.QFont()
        font.setFamily('Monospace')
        self.textbox.setFont(font)
        vbox.addWidget(self.textbox)

        self.refresh()
        self.setLayout(vbox)
        self.setWindowTitle('State: {} arm'.format(parent.arm_name))
        self.setGeometry(100, 600, 1500, 250)
        self.show()

    def closeEvent(self, event):
        self.parent.state_popup = None
        event.accept()

    def refresh(self):
        state = self.parent.arm.get_state()
        keys = sorted(k for k in state.keys() if k not in self.ignore)
        pad = max(len(k) for k in keys) + 2
        values = [state[k].tolist() if isinstance(state[k], np.ndarray) else state[k] for k in keys]
        state_text = '\n'.join('{}: {}'.format(k.ljust(pad), v) for k, v in zip(keys, values))
        self.textbox.setText(state_text)

def center(widget):
    widget.setSizePolicy(qtg.QSizePolicy.Expanding, qtg.QSizePolicy.Preferred)
    widget.setAlignment(qtc.Qt.AlignCenter)

class BackgroundWorker(qtc.QThread):
    def __init__(self, slider_window, action):
        super(BackgroundWorker, self).__init__(slider_window)
        self.do_action = action
        slider_window.set_moving(True)
        self.finished.connect(lambda: slider_window.set_moving(False))
        self.start()
    def run(self):
        self.do_action()

class SliderWindow(qtg.QWidget):
    def __init__(self, arm_name):
        super(SliderWindow, self).__init__()
        self.arm_name = arm_name
        self.arm = BaxterArm(arm_name)

        vbox = qtg.QVBoxLayout()
        header_box = qtg.QHBoxLayout()
        header_box2 = qtg.QHBoxLayout()
        sliders_box = qtg.QHBoxLayout()
        vbox.addLayout(header_box)
        vbox.addLayout(header_box2)
        vbox.addLayout(sliders_box)

        tuck = qtg.QPushButton('tuck')
        tuck.clicked.connect(self.handle_tuck)
        header_box.addWidget(tuck)

        tuck = qtg.QPushButton('untuck')
        tuck.clicked.connect(self.handle_untuck)
        header_box.addWidget(tuck)

        state_popup = qtg.QPushButton('state')
        state_popup.clicked.connect(self.handle_state_popup)
        header_box.addWidget(state_popup)
        self.state_popup = None

        self.moving_status = qtg.QLabel('')
        self.moving_status.setMargin(1)
        self.moving_status.setStyleSheet('color: red')
        self.moving = False
        header_box.addWidget(self.moving_status)

        # note: doesn't apply to tuck/untuck
        header_box2.addWidget(qtg.QLabel('timeout:'))
        self.timeout_spinbox = qtg.QDoubleSpinBox()
        self.timeout_spinbox.setRange(1, 15)
        self.timeout_spinbox.setSingleStep(0.1)
        # default for AML: 15
        self.timeout_spinbox.setValue(5)
        self.timeout_spinbox.setDecimals(1)
        header_box2.addWidget(self.timeout_spinbox, 2) # stretch factor

        # note: doesn't apply to tuck/untuck
        header_box2.addWidget(qtg.QLabel('threshold:'))
        self.threshold_spinbox = qtg.QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.005, 0.1)
        self.threshold_spinbox.setSingleStep(0.001)
        # default threshold for AML = 0.008726646
        self.threshold_spinbox.setValue(0.05)
        self.threshold_spinbox.setDecimals(3)
        header_box2.addWidget(self.threshold_spinbox, 2) # stretch factor

        self.joint_names = self.arm.joint_names()
        self.num_joints = len(self.joint_names) # 7
        self.joint_sliders = []
        self.joint_slider_labels = []
        for i, name in enumerate(self.joint_names):
            limits = self.arm._jnt_limits[i]

            slider_vbox = qtg.QVBoxLayout()
            description = my_descriptions[name.split('_')[1]]
            name_label = qtg.QLabel('{}\n{}'.format(name, description))
            center(name_label)
            slider_vbox.addWidget(name_label)
            slider = FloatSlider(qtc.Qt.Vertical)
            self.joint_sliders.append(slider)
            label = qtg.QLabel()
            center(label)
            self.joint_slider_labels.append(label)

            slider.set_float_range(limits['lower'], limits['upper'])
            slider.sliderMoved.connect(self.handle_slider_moved)
            slider.sliderReleased.connect(self.handle_slider_released)

            b = qtg.QHBoxLayout()
            b.addWidget(slider)
            b.setAlignment(qtc.Qt.AlignHCenter)
            slider_vbox.addLayout(b)

            slider_vbox.addWidget(label)
            sliders_box.addLayout(slider_vbox)
        self.sync_sliders()

        timer = qtc.QTimer(self)
        timer.setInterval(100) # ms
        timer.timeout.connect(self.sync_sliders)
        timer.start()

        self.setLayout(vbox)
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle('Joint Sliders: {} arm'.format(arm_name))
        self.setWindowFlags(qtc.Qt.WindowStaysOnTopHint)
        self.show()

    def set_moving(self, moving):
        self.moving = moving
        self.moving_status.setText('moving' if moving else '')
        if moving:
            qtg.QApplication.setOverrideCursor(qtc.Qt.WaitCursor)
            # sometimes some events are missed the first time. Not sure why...
            qtg.QApplication.processEvents()
            time.sleep(0.05)
            qtg.QApplication.processEvents()
        else:
            qtg.QApplication.restoreOverrideCursor()
            self.sync_sliders()

    def handle_tuck(self):
        BackgroundWorker(self, self.arm.tuck_arm)
    def handle_untuck(self):
        BackgroundWorker(self, self.arm.untuck_arm)
    def handle_joint_command(self, joint_command, timeout, threshold):
        BackgroundWorker(self, lambda:
            self.arm.move_to_joint_positions(joint_command, timeout=timeout, threshold=threshold))

    def handle_state_popup(self):
        if self.state_popup is None:
            self.state_popup = StatePopup(self)

    def handle_slider_released(self):
        ''' when the slider is released, move the robot to that configuration '''
        if self.moving:
            return # don't process when moving
        joint_positions = [j.get_float_value() for j in self.joint_sliders]
        for i, l in enumerate(self.joint_slider_labels):
            l.setText('{:.3f}'.format(joint_positions[i]))

        joint_command = dict(zip(self.joint_names, joint_positions))
        timeout = self.timeout_spinbox.value()
        threshold = self.threshold_spinbox.value()
        self.handle_joint_command(joint_command, timeout, threshold)

    def handle_slider_moved(self):
        ''' update the slider labels
        triggered when the user is dragging, but not when the value changes for other reasons
        '''
        for i in range(self.num_joints):
            val = self.joint_sliders[i].get_float_value()
            self.joint_slider_labels[i].setText('{:.3f}'.format(val))

    def sync_sliders(self):
        ''' update the slider and slider labels based on the robot configuration '''
        joint_positions = self.arm.get_state()['position']
        for i in range(self.num_joints):
            val = joint_positions[i]
            # don't want to disrupt the user if they are currently dragging a slider
            if not self.joint_sliders[i].isSliderDown():
                self.joint_sliders[i].set_float_value(val)
                self.joint_slider_labels[i].setText('{:.3f}'.format(val))

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['left', 'right']:
        print('usage: ./joint_sliders.py (left|right)')
        sys.exit(1)

    arm_name = sys.argv[1]

    rospy.init_node('joint_sliders')
    ros_thread = threading.Thread(target=rospy.spin)
    ros_thread.daemon = True
    ros_thread.start()

    # this will make Ctrl+C will close pyQt (https://stackoverflow.com/a/5160720)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    # in the container C.UTF-8 by default which causes 'Fontconfig warning: ignoring C.UTF-8: not a valid language tag'
    os.environ['LC_ALL'] = 'C'
    app = qtg.QApplication(sys.argv)
    win = SliderWindow(arm_name)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

