# Thesis
Point base value iteration of zero sum stackelberg POSGs

to run a game use the below command :

python test.py <file_name> <game_type> <horizon> <iterations> <sota(1 or 0)>

example :
python test.py problem=dectiger gametype=cooperative horizon=3 iter=3 sota=0


filename should be "dectiger"
where gametype is a string from :{ "cooperative" , "stackelberg" or "zerosum"

current version's SOTA for zerosum does not work
