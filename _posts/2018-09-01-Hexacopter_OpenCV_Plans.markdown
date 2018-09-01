---
layout: post
title:  "Hexacopter, Software Setup, OpenCV + Future Plans"
date:   2018-09-01 23:42:55 -0400
categories: jekyll update
---
Currently the Hexacopter is running on APM2.6. We can connect to the Raspberry Pi over a WiFi network on port UDP:14553 to actively see Mission Planner Data.

1. The next step here would be to figure out how to stream data using the predefined functions from the Navio2 Github to build an online Dashboard. Would attempt to use Pandas, a websocket, and Bokeh or Plotly to build this.

We were also able to successfully mount a version of openCV onto the Hexacopter. Learning how to do object detection and later expanding that to controlling the drone either through ROS or Dronekit's API would be the best strategy for Search and Rescue Style Missions.
