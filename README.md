# Lighthouses AI Contest "Reloaded"

Lighthouses AI Contest is a turn based game built by [Hector Martin aka "marcan"](https://github.com/marcan/lighthouses_aicontest), as the challenge for the AI contest within one of the largest and oldest demoparty and LAN party, the [Euskal Encounter](https://ee32.euskalencounter.org/) in Bilbao, Spain.

I have updated from his original Python 2.7 code, to a Go 1.22 languaje and also I have added a few more changes:

- Original version
  - Python 2.7
  - PyGame for visualization
  - Based on comunications via stdin / stdout
  - Maps esigned for a few players at the same game
  - Built to run in a stand alone computer
- My "Reloaded" version
  - Go 1.22
  - No Visualization layer
  - Based on TCP communication
  - Focused on allowing many players in the same game
  - Container first
