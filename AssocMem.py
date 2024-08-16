import sys
import os
import numpy as np
import cupy as cp
import cv2
import matplotlib.pyplot as plt

class AssocMem:
    def __init__(self, image_name, p, seed=1234):
        """
        Initializes the associative memory with the given parameters.

        Args:
            image_name (str): The name of the image file to be used as the pattern.
            p (int): The number of patterns to memorize.
            seed (int, optional): The random seed for reproducibility. Defaults to 1234.
        """
        self.image_name = image_name
        self.p = p
        self.m0 = 0
        self.delta_m = 0
        self.max_steps = 0
        self.seed = seed
        self.s_list = []
        self.m = []

        self.image = cv2.imread(self.image_name, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            print(f"[Error] Cannot open \"{self.image_name}\"")
            sys.exit(-1)
        self.xi0 = cp.where(cp.asarray(self.image.reshape(-1)) == 255, 1, -1)
        self.n = self.xi0.size
        self.width = self.image[0].size
        self.height = self.n // self.width

    def memorize(self):
        """
        Memorizes the pattern using the Hopfield network.
        Generates a synaptic matrix J based on the input patterns.
        """
        cp.random.seed(np.uint64(self.seed))
        xi = cp.random.choice([-1, 1], size=(self.p - 1, self.n))
        self.J = (cp.outer(self.xi0, self.xi0) + xi.T @ xi) / self.n

    def recall(self, m0, delta_m=0.001, max_steps=100):
        """
        Recalls the pattern from memory using the Hopfield network.
        Simulates the network dynamics for a maximum of `max_steps` or until convergence.
        Stores each state of s in `s_list` and calculates m at each step.

        Args:
            m0 (float): The initial overlap between the input and the memorized pattern.
            delta_m (float, optional): The convergence threshold for m0. Defaults to 0.001.
            max_steps (int, optional): The maximum number of recall steps. Defaults to 100.
        """
        self.m0 = m0
        self.delta_m = delta_m
        self.max_steps = max_steps
        s = self.xi0.copy()
        indices = cp.random.choice(self.n, size=int(self.n * (1 - self.m0) / 2), replace=False)
        s[indices] = -s[indices]
        self.m = [float(cp.dot(self.xi0, s) / self.n)]
        self.s_list = [s.get()]

        for _ in range(self.max_steps):
            s = cp.where((self.J @ s) >= 0, 1, -1)
            self.s_list.append(s.get())
            self.m.append(float(cp.dot(self.xi0, s) / self.n))
            if cp.abs(self.m[-1] - self.m[-2]) <= self.delta_m:
                break

    def plot_m(self, save_name=None, dpi=300):
        """
        Plots the similarity m as a function of the recall steps.

        Args:
            save_name (str, optional): If provided, saves the plot to this file. Defaults to None.
            dpi (int, optional): The resolution in dots per inch of the saved figure. Defaults to 300.
        """
        fig, ax = plt.subplots()
        ax.grid()
        ax.set_title(f'Number of patterns = {self.p}, $m^0(t=0) = ${self.m0}')
        ax.plot(np.arange(len(self.m)), self.m, 'ro')
        ax.set_xlabel('Steps')
        ax.set_ylabel('$m^0$')
        ax.set_ylim(top=1.01)
        if save_name is not None:
            fig.savefig(save_name, dpi=dpi)
        plt.show()

    def save_video(self, output_name=None, output_size=(1920, 1080)):
        """
        Saves the recall process as a video.

        Args:
            output_name (str, optional): The name of the output video file. Defaults to None.
            output_size (tuple, optional): The size of the output video as (width, height). Defaults to (1920, 1080).
        """
        if output_name is None:
            output_name = f'{os.path.splitext(os.path.basename(self.image_name))[0]}_P_{self.p}_M0_{self.m0}.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_name, fourcc, 1, output_size, isColor=False)
        if not video.isOpened():
            print(f"[Error] Cannot open \"{output_name}\"")
            sys.exit(-1)

        for s in self.s_list:
            frame = self._resize_to_fhd(np.where(s.reshape((self.height, self.width)) == 1, 255, 0).astype(np.uint8), output_size)
            video.write(frame)
        
        video.release()

    def free(self):
        """
        Frees the GPU memory used by the class.
        Resets all class variables and releases the memory pools.

        Returns:
            tuple: A tuple containing used_bytes, total_bytes, and n_free_blocks from the memory pool of cupy.
        """
        self.xi = None
        self.xi0 = None
        self.s = None
        self.J = None
        self.indices = None
        self.s_list.clear()
        self.m.clear()
        self.image = None

        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        return mempool.used_bytes(), mempool.total_bytes(), pinned_mempool.n_free_blocks()

    def _resize_to_fhd(self, image, output_size):
        """
        Resizes an image to the specified size while maintaining the aspect ratio.

        Args:
            image (numpy.ndarray): The input image to resize.
            output_size (tuple): The desired output size as (width, height).

        Returns:
            numpy.ndarray: The resized image with the specified output size.
        """
        scale = min(output_size[0] / self.width, output_size[1] / self.height)
        new_width = int(self.width * scale)
        new_height = int(self.height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        fhd_image = np.zeros((output_size[1], output_size[0]), dtype=np.uint8)

        x_offset = (output_size[0] - new_width) // 2
        y_offset = (output_size[1] - new_height) // 2
        fhd_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image

        return fhd_image
