import cv2
import imageio
import numpy as np

import sys
sys.path.append('/workspace/S/heguanhua2/robot_rl/robosuite_jimu')
import robosuite as suite
import robosuite.macros as macros


# import robosuite as suite
# import robosuite.macros as macros

macros.IMAGE_CONVENTION = "opencv"

# env = suite.make("Lift", robots="Panda", camera_names="frontview")
# obs = env.reset()
# img = obs["frontview_image"]
# low, high = env.action_spec
# action = np.random.uniform(low, high)
# obs, reward, done, env_info = env.step(action)
# print(f"obs keys=\n{obs.keys()}",
#       f"reward={reward}",
#       f"done={done}",
#       f"env_info=\n{env_info}")
# writer = imageio.get_writer("./test_video.mp4", fps=10)
# writer.append_data(img)
# writer.close()
# imageio.imwrite("./test_img.png", img)

camera = "frontview"

env = suite.make(
    "Jimu",
    "UR5e",
    has_renderer=False,
    ignore_done=True,
    use_camera_obs=True,
    use_object_obs=False,
    camera_names=camera,
    camera_heights=256,
    camera_widths=256,
)

print('make success')

obs = env.reset()
ndim = env.action_dim
low, high = env.action_spec
print(f"ndim = {ndim}")
print(f"low, high = {low}, {high}")

# create a video writer with imageio
writer = imageio.get_writer("./jimu_video.mp4", fps=20)

frames = []
for i in range(200):

    # run a uniformly random agent
    action = 0.5 * np.random.randn(ndim)
    obs, reward, done, info = env.step(action)

    # dump a frame from every K frames
    if i % 1 == 0:
        frame = obs[camera + "_image"]
        writer.append_data(frame)
        # print("Saving frame #{}".format(i))

    if done:
        break

writer.close()
