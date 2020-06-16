import argparse

import numpy as np
from path import Path

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Options for trajectory visualization.')
parser.add_argument('--kitti', type=str, help='visualize KITTI ground truth trajectory', default="")
parser.add_argument('--homework', type=str, help='visualize HomeworkVO trajectory', default="")
args = parser.parse_args()


def main():
  kitti_data = np.loadtxt(Path(args.kitti))
  kitti_data = np.reshape(kitti_data, (kitti_data.shape[0], 3, 4))
  kitti_x = kitti_data[:50,0,3]
  kitti_y = kitti_data[:50,2,3]

  homework_data = np.loadtxt(Path(args.homework))
  homework_data = np.reshape(homework_data, (homework_data.shape[0], 3, 4))
  homework_x = homework_data[:,0,3]
  homework_y = homework_data[:,2,3]

  plt.figure()

  plt.plot(kitti_x, kitti_y, label="GT", linewidth=1)
  plt.plot(homework_x, homework_y, label="HomeworkVO", linewidth=1)

  plt.axis('square')
  plt.legend()
  plt.savefig("trajectory.svg", bbox_inches='tight')


if __name__ == "__main__":
  main()