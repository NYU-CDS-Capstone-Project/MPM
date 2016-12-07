import shlex, subprocess

def form_simulator_command(n_event=25000, sqrtshalf=45, polbeam1=0, polbeam2=0):
	simulator_command = "./run.sh {} {} {} {}".format(n_event, sqrtshalf, polbeam1, polbeam2)
	return simulator_command

subprocess.call(form_simulator_command(), shell=True)