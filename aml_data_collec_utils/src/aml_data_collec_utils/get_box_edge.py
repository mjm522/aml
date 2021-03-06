import numpy as np


def get_box_edge2(u_alpha, bw, bh):

	"""

	bw x bh
			s=0 [0,0.25]
		a0 ---- a1
		|       |
	s=3	|   c   | s=1, (0.25,0.50]
		|       |
		a3 ---- a2
		   s=2, (0.5,0.75]
	"""

	assert(u_alpha >= 0 and u_alpha <= 1.0)


	a0 = 0.0
	a1 = 0.25
	a2 = 0.50
	a3 = 0.75

	p0 = np.array([-bw,-bh])
	p1 = np.array([bw,-bh])
	p2 = np.array([bw,bh])
	p3 = np.array([-bw,bh])

	p = None

	pts = np.vstack([p0,p1,p2,p3,p0])#*0.5


	ai = [a0,a1,a2,a3,1.0]

	sides = [3, 1, 4, 2]


	alpha = 0.0
	sidx = 0
	if 0. < u_alpha <= 0.25:
		sidx = 0
		alpha = u_alpha/0.25
		p = (1.0 - alpha)*p0 + alpha*p1

	elif 0.25 < u_alpha <= 0.5:
		sidx = 1
		alpha = (u_alpha-0.25)/0.25
		p = (1.0 - alpha)*p2 + alpha*p1

	elif 0.5 < u_alpha <= 0.75:
		sidx = 2
		alpha = (u_alpha-0.5)/0.25

		p = (1.0 - alpha)*p2 + alpha*p3

	elif 0.75 < u_alpha < 1.0:
		sidx = 3
		alpha = (u_alpha-0.75)/0.25

		p = (1.0 - alpha)*p0 + alpha*p3

	# elif 0.25 < u_alpha <= 0.5:
	# 	side = 1
	# 	alpha = (u_alpha-0.25)/0.25
	# elif 0.5 < u_alpha <= 0.75:
	# 	side = 2
	# 	alpha = (u_alpha-0.5)/0.25
	# elif 0.75 < u_alpha <= 1.:
	# 	side = 3
	# 	alpha = (u_alpha-0.75)/0.25


	# alpha = 0.0
	# side = 0

	# while side < len(ai) and u_alpha > ai[side]:
	# 	side += 1


	# alpha = (u_alpha - ai[sidx-1])/(ai[sidx] - ai[sidx-1])


	# p = (1.0 - alpha)*pts[sides[sidx],:] + alpha*pts[sides[sidx]-1,:]



	return p, sides[sidx]

def get_box_edge(u_alpha, bw, bh):

	p, _ = get_box_edge2(u_alpha, bw, bh)



	return p


def main():


	import matplotlib.pyplot as plt

	u_alphas = np.linspace(0,1,200)
	print "u_alphas: ", u_alphas


	bpts = np.vstack([ get_box_edge(u,10,10) for u in u_alphas ])

	print "Box pts: ", bpts


	plt.plot(bpts[:,0],bpts[:,1])
	plt.show()


if __name__ == '__main__':
	main()