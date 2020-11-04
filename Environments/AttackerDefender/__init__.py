from gym.envs.registration import register

register(
	id = 'AttackerDefender-v0',
	entry_point = 'Environments.AttackerDefender.envs:ADFL'
	)

#The id variable we enter here is what we will pass into gym.make() to call our environment.