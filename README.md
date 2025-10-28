# pibt_rip

poetry run python app.py -m assets/random-32-32-10.map -i assets/random-32-32-10-random-1.scen -N 200
mapf-visualizer ./assets/random-32-32-10.map ./output.txt

poetry run python3 app.py -m assets/pushmap.map -i assets/pushmap-random-1.scen -N 4
mapf-visualizer ./assets/pushmap.map ./output.txt


poetry run python app.py -m assets/small.map -i assets/small-random-1.scen -N 2
mapf-visualizer ./assets/small.map ./output.txt

poetry run python app.py -m assets/tunnel.map -i assets/tunnel.scen -N 4
mapf-visualizer ./assets/tunnel.map ./output.txt


poetry run python app.py -m assets/maze-128-128-1.map -i assets/scen/maze-128-128-1-random-1.scen -N 200
mapf-visualizer ./assets/maze-128-128-1.map ./output.txt

todo:
poetry run python3 app.py -m assets/corswap.map -i assets/corswap.scen -N 10
mapf-visualizer ./assets/corswap.map ./output.txt

poetry run python app.py -m assets/random-32-32-20.map -i assets/scen/random-32-32-20-random-4.scen -N 200
mapf-visualizer ./assets/random-32-32-20.map ./output.txt

fail~
poetry run python app.py -m assets/random-32-32-20.map -i assets/scen/random-32-32-20-random-5.scen -N 200

poetry run python app.py -m assets/random-32-32-20.map -i assets/scen/random-32-32-20-random-15.scen -N 100

poetry run python app.py -m assets/warehouse-20-40-10-2-1.map -i assets/scen/warehouse-20-40-10-2-1-random-1.scen -N 50
