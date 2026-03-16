from ._base_task import Base_Task
from .utils import *
import sapien
import math

class test_gripper(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        pass
    
    def reach_test(self, arm_tag):
        if arm_tag == 'left':
            init_pose = sapien.Pose([-0.252007, -0.50, 0.975889], [0.707054, -0.00149809, 0.00148721, 0.707156])
        else:
            init_pose = sapien.Pose([0.252007, -0.50, 0.975889], [0.707054, -0.00149809, 0.00148721, 0.707156])
        box = add_robot_visual_box(self, init_pose, 'target')
        res_box, ee_target_box, ee_box, real_ee_box = None, None, None, None
        while True:
            target_pose = box.get_pose()
            if not np.allclose(target_pose.p, init_pose.p, atol=0.001) \
                or not np.allclose(target_pose.q, init_pose.q, atol=0.001):
                init_pose = target_pose

                contact_matrix = transforms._toPose(target_pose).to_transformation_matrix()
                global_contact_pose_matrix_q = contact_matrix[:3, :3]
                global_grasp_pose_p = (contact_matrix[:3, 3] +
                                    global_contact_pose_matrix_q @ np.array([-0.12, 0, 0]).T)
                global_grasp_pose_q = t3d.quaternions.mat2quat(global_contact_pose_matrix_q)
                res_pose = list(global_grasp_pose_p) + list(global_grasp_pose_q)
                self.move(self.move_to_pose(arm_tag, res_pose))

                if not self.plan_success:
                    print('plan failed!')

                res_pose_pose = transforms._toPose(res_pose)
                if res_box is None:
                    res_box = add_robot_visual_box(self, res_pose_pose, 'res')
                res_box.set_pose(res_pose_pose)
                res_box.set_name('success' if self.plan_success else 'fail')

                ee_target_pose = transforms._toPose(
                    self.robot._trans_from_gripper_to_endlink(res_pose, arm_tag=arm_tag))
                if ee_target_box is None:
                    ee_target_box = add_robot_visual_box(self, ee_target_pose, 'ee_target')
                ee_target_box.set_pose(ee_target_pose)
                
                ee_pose = transforms._toPose(
                    self.robot.get_left_ee_pose() if arm_tag == 'left' else self.robot.get_right_ee_pose())
                if ee_box is None:
                    ee_box = add_robot_visual_box(self, ee_pose, 'ee')
                ee_box.set_pose(ee_pose)

                ee_bias = ee_target_pose.p - ee_pose.p
                print('frame bias(ee):', np.round(ee_bias, 4).tolist())
 
                if arm_tag == 'left':
                    real_ee_pose = self.robot.left_ee.get_global_pose()
                else:
                    real_ee_pose = self.robot.right_ee.get_global_pose()
                real_ee_pose = transforms._toPose(real_ee_pose)
                if real_ee_box is None:
                    real_ee_box = add_robot_visual_box(self, real_ee_pose, 'real_ee')
                real_ee_box.set_pose(real_ee_pose)

                self.plan_success = True
            self.viewer.render()
    
    def grasp_test(self, arm_tag, pre_dis=0.1):
        shoes_pose = sapien.Pose([0.35, -0.12, 0.75], [0.707, 0.707, 0, 0])
        self.shoe_id = np.random.choice([i for i in range(10)])
        self.shoe = create_actor(
            scene=self,
            pose=shoes_pose,
            modelname="041_shoe",
            convex=True,
            model_id=self.shoe_id,
        )

        pose_list = [
            sapien.Pose([0.35, -0.12, 0.75], [0.707, 0.707, 0, 0]),
            sapien.Pose([0.32, -0.12, 0.75], [0.707, 0.707, 0, 0]),
            sapien.Pose([0.3, -0.12, 0.75], [0.707, 0.707, 0, 0])
        ]
        
        # pose_list = []
        # for x in np.linspace(0.35, 0.3, 10):
        #     for y in np.linspace(-0.12, -0.12, 1):
        #         pose_list.append(sapien.Pose([x, y, 0.75], [0.707, 0.707, 0, 0]))
 
        self.viewer.render()
        for pose in pose_list:
            self.move(self.back_to_origin(arm_tag=arm_tag))
            self.move(self.open_gripper(arm_tag=arm_tag))
            self.shoe.actor.set_pose(pose)
            
            try:
                grasp_action = self.grasp_actor(self.shoe, arm_tag=arm_tag, pre_grasp_dis=pre_dis, gripper_pos=0)
                self.move(grasp_action)
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=pre_dis, move_axis='arm'))
            except Exception as e:
                self.plan_success = False

            print(f'Pose: {pose.p}, result: {self.plan_success}')
            self.viewer.render()
            self.plan_success = True
        pause(self)

    def play_once(self):
        # self.grasp_test('right', pre_dis=0.1)
        self.reach_test('left')
        
        self.info["info"] = {
            "{A}": f"001_bottle/base0",
            "{a}": 'left',
        }
        return self.info

    def check_success(self):
        return True