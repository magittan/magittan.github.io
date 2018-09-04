---
layout: post
title:  "Plasmon Simulation Project"
date:   2018-09-01 23:42:55 -0400
categories: jekyll update
---

This blog post is going to be about solving wave equations! Specifically the wave-equations that describe a plasmon, which is a quantized plasma oscillation. Solving the wave equation of the plasmon on a sample, with sources and reflectors placed arbitrarily, allows us to visualize plasmons in complex geometries.

In order to solve the wave-equation for plasmons on a specified mesh we must employ FEniCS. According to their website FEniCS is, "an open-source computing platform for solving partial differential equations (PDEs)."

The basic approach to solving a problem in FEniCS is to first define the mesh and subdomains, specifying the boundary conditions, then specifying the solver to solve the problem.

The following brief overview is substantiated with the following brief manuscript [here](/static/Plasmon_Simulation_Project/modeling-plasmons.pdf).
