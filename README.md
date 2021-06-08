# LSS

In this repo, I have included the training codes, generated weights, and Jupyter notebooks for four systems:

<ol>
<li>BBA</li>
<li>Villin</li>
<li>PRB</li>
<li>AT-All DNA</li>
</ol>

The training paradigm with the best results are marked accordingly before the directory name.

The **BBA** and **Villin** systems were trained to completion. As of writing this, **PRB** and **DNA** still have some issues.

**BBA** was trained with the <em>DESRES-Trajectory_1FME-1-protein</em> trajectories.
**Villin** was trained with the <em>DESRES-Trajectory_2F4K-0-protein</em> trajectories.
**PRB** was trained with the <em>DESRES-Trajectory_2F4K-0-protein</em> trajectories.

**DNA** is trained using files hosted on Midway, located at <em>/project2/andrewferguson/mikejones/AT-all_SRV_data</em>. For DNA, the system is trained using Mike's SRV coordinates, and the corresponding all atom positions. The sub directories of <em>/project2/andrewferguson/mikejones/AT-all_SRV_data</em> contain the individual trajectories as well as SRV coordinates. Also, the training python file for the DNA system should be clear on which directories of the DNA files correspond to which files.
