###Setup `configure` on Mac
1. Download and open the Docker for Mac App
2. Open Utilities/XQuartz (download/update if using for first time.)
3.	Type `xhost + 127.0.0.1` in a terminal
4.	Type `docker run -e DISPLAY=host.docker.internal:0 -it -v "$(pwd)":/src ubuntu` into the terminal	
5.	Go into the `/src` directory in the terminal and you should see all your files.
6.	Type `apt-get -y update && apt-get install libfontconfig libxft-dev` to download the packages configure needs
7. `cd configure-8.4-linux-intel-64bit`
8.	Launch `./configure`


###GW170817
* Tile 0: Looks good.
* Tile 1: Looks good.
* Tile 2: Looks good.
* Tile 3: Looks good.
* Tile 4: Looks good.
* Tile 5: Looks okay. 6 fiducials allocated on either Plate.
* Tile 6: Looks good.
* Tile 7: Looks good.

###GW190814
* Tile 0: Looks good.
* Tile 1: Looks good.
* Tile 2: Could do with one more fiducial in the north-east. Only 5 fiducials on Plate 0. 6 allocated on Plate 1, so make sure to configure on Plate 1 and should be fine.
* Tile 3: Looks good.
* Tile 4: Looks good.
* Tile 5: Looks good.
* Tile 6: Looks good.
* Tile 7: Looks good.
* Tile 8: Looks good.
* Tile 9: Looks good.

###GW200129
* Tile 0: Looks good.
* Tile 1: Looks good.
* Tile 2: Looks good.
* Tile 3: Looks good.
 