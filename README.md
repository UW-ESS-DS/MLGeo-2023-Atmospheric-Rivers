# MLGeo-2023-Atmospheric-Rivers
Project group interested in predicting atmospheric river strength and temperature characteristics from ERA5 reanalysis

### Group members
**ESS 569 Students**
- Chad Small
- Danny Hogan

**ESS 469 Students**
- Hamzi Rapi

  # Background
  Along the west coast of North America, total precipitation over the year is often controlled by a relatively small number of individual storm events. Many of these events fall into the category of atmospheric rivers (ARs), defined as long, narrow bands of enhanced moisture entrained from the tropics that are then advected into the mid-latitudes (Dettinger et al., 2011; Rutz et al., 2014; Rutz & Steenburgh, 2012). These events lead to significant amounts of precipitation at the surface. In some cases, these deluges can be beneficial to stem droughts and bolster water resources (Zhu and Newell, 1988; Guan and Waliser, 2015). ARs can also be destructive, causing flooding or mass wastage events that impact communities and damage infrastructure. Thus, it is vital to predict the severity of these events with accuracy to better prepare a region for the potential impacts from these storms.
 An AR scale was recently developed to provide an intensity factor for a given event (Ralph et al., 2019). This factor takes the modeled duration of the AR and the maximum modeled integrated vapor transport (IVT)  into account. The duration metric is used to estimate the amount of time the AR would lie over a particular location. IVT is a measure of the “flow rate” of the AR, similar to the flow rate of the river. Essentially, it is a measure of the flux of water vapor integrated over the atmospheric column to capture both how much and how fast moisture is moving in the AR. 
The scale ranges between 1, a relatively short duration event (less than 24 hours) with low IVT, to 5, a longer duration event (36 hours or more) with high IVT (above 1000 kg/m2/s) (Figure 1). While this scale is useful to simply communicate the general severity of these storms, we believe it has two apparent weaknesses. (1) The classification does not take prior events into account, which may exacerbate the effects of a relatively weaker AR, and (2) the classification does not describe precipitation phases at higher elevations, where precipitation rates will often be larger due to orographic enhancement from surrounding terrain. In terrain where snow regularly falls (>1000 m in Washington state), the precipitation phase of an AR plays an important role in controlling whether water from a given storm will be stored in the snow pack, or runoff more quickly into surrounding rivers. The former example would bolster water resources for the drier months, while the latter example poses greater risk for more severe flooding impacts downstream.
	We sought to address the second of these two AR classification problems by using data from meteorological reanalysis to group recent atmospheric river events into classes based on both precipitation phase and storm intensity. 

# Organization
All notebooks needed to produce results are contained within the `notebooks` folder. Project data is organized and contained within the `Project Data` folder.
