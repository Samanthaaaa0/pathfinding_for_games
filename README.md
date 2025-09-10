# pibt_rip

poetry run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 200
mapf-visualizer ./assets/random-32-32-10.map ./output.txt

poetry run python3 app.py -m assets/pushmap.map -i assets/pushmap-random-1.scen -N 4
mapf-visualizer ./assets/pushmap.map ./output.txt

poetry run python app.py -m assets/small.map -i assets/small-random-1.scen -N 2
mapf-visualizer ./assets/small.map ./output.txt

poetry run python app.py -m assets/smallpush.map -i assets/smallpush-random-1.scen -N 2
mapf-visualizer ./assets/smallpush.map ./output.txt

poetry run python3 app.py -m assets/tcross.map -i assets/tcross-random-1.scen -N 4
mapf-visualizer ./assets/tcross.map ./output.t
poetry run python app.py -m assets/maze-128-128-1.map -i assets/scen-random/maze-128-128-1-random-1.scen -N 200
mapf-visualizer ./assets/maze-128-128-1.map ./output.txt


--- for standard pibt ---
poetry run python3 app.py -m assets/pushmap.map -i assets/pushmap-random-1.scen -N 4


poetry run python app.py -m assets/random-32-32-20.map -i assets/32-32-20-scen/random-32-32-20-random-4.scen -N 200
mapf-visualizer ./assets/random-32-32-20.map ./output.txt

