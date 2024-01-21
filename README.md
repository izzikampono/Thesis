# Thesis
Point base value iteration of zero sum stackelberg POSGs

to run a test game (with option to print policy trees)  use the below command :

python test.py <file_name> <game_type> <horizon> <iterations> <sota(1 or 0)>
example :
python test.py problem=dectiger gametype=cooperative horizon=3 iter=3 sota=0

where gametype is a string from : { "cooperative" , "stackelberg" or "zerosum"}

to run a while experiment:
python test.py <file_name>  <horizon> <iterations>

example :
python experiment.py problem=dectiger horizon=10 iter=10

this repository also include a job.sh bash script to run on an ssh server.
the job.sh file take the name of the game as a command line argument,

example:
 sbatch job.sh tiger


