"""
    Image noising module.
"""
import math
import numpy as np
import numpy.random


def noisy(image, noise_type):
	"""
		Add noise to an image.

		image: Input image, ndarray, uint8 type.
		noise_type: One of the following types:
		'gaussian': Gaussian-distributed additive noise.
		'salt_and_pepper':	Salt and pepper noise, replacing random pixels
					with maximal value (salt) or minimal value (pepper).
		'poisson': 	Poisson-distributed noise generated from the data.
		'speckle': 	Multiplicative noise. output = image + image * n, where
					n is uniform noise with specified mean & variance.
	"""
	if noise_type == 'gaussian':
		mean = 0
		sigma = 10
		gauss = np.random.normal(mean, sigma, image.shape)
		output = image + gauss
	elif noise_type == 'salt_and_pepper':
		amount = 0.04
		salt_vs_pepper = 0.5
		output = np.copy(image)

		# Salt mode
		n_pixel_salt = int(image.size * amount * salt_vs_pepper)
		coords = [np.random.randint(0, i - 1, int(n_pixel_salt)) for i in image.shape]
		output[coords] = np.iinfo(image.dtype).max

		# Pepper mode
		n_pixel_pepper = int(image.size * amount * (1 - salt_vs_pepper))
		coords = [np.random.randint(0, i - 1, int(n_pixel_pepper)) for i in image.shape]
		output[coords] = np.iinfo(image.dtype).min
	elif noise_type == 'poisson':
		vals = len(np.unique(image))
		vals = pow(2, math.ceil(math.log2(vals)))
		output = np.random.poisson(image * vals) / float(vals)
	elif noise_type == 'speckle':
		mean = 0
		sigma = 0.05
		gauss = np.random.normal(mean, sigma, image.shape)
		output = image + image * gauss
	else:
		print('Error: unexpected noise type!')
		return
	return output
