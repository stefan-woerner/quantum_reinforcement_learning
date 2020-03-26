from gym.envs.registration import register

register(
	id = 'FL-v1',
	entry_point = 'Environments.FL.envs:FL'
	)

#The id variable we enter here is what we will pass into gym.make() to call our environment.