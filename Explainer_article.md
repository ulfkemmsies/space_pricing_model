![](https://cdn.mathpix.com/snip/images/jpjHzOgP2RAMq9b10NlOVgPUK_6coHC1x1alJKnlScY.original.fullsize.png)
- # Introduction
	- The commercial space industry, which has flourished in this last decade due to rocket reusability, satellite miniaturization, and wider data analysis capabilities, is slowly moving outward from Earth orbit to the greater cislunar space which envelops our home. A longstanding narrative is that of exploiting space resources to benefit our techno-industrial complex, eventually perhaps even substituting resources from our own biosphere in order to salvage it. In cislunar space (and a bit beyond), lunar and Near Earth Object (NEO) resources are the easiest to reach and the probable first targets for exploration. This process can be greatly cheapened by transport infrastructure at key locations, and conversely, the harvested construction materials can cheapen the construction of such infrastructure.
	- A possible metric for the progress of cislunar development is the price of goods at any given location: comparing goods coming from different sources (NEOs, Moon, Earth) shows how efficient the overall system is for that specific trajectory and where more infrastructure is needed. In order to forecast these prices, orbital mechanics, vehicle parameters, and rocket equations must be taken into account. Unlike logistics on Earth, where shipping costs are a fraction of the final price paid, space transport is still so difficult and expensive that is remains the main cost driver.
	- This article is the first step in an attempt to create **a framework, model, and software implementation to estimate how much space resources will cost at different orbital points of interest**. It will be written for both laypeople and industry members using expandable content for the underlying equation derivations and deeper detail. It will also accompany a [simple app](https://share.streamlit.io/ulfkemmsies/space_pricing_model/main) that implements the theory. Many assumptions are being made herein, and since they will vary in their quality, this article is also an attempt to validate them through the feedback it will generate from knowledgeable readers.
	- The main steps taken herein are as follows:
		- Define all background information, equations and terms
		- Define the main cost drivers for idealized space cargo transport
		- Introduce simplifications (mass ratios) that mitigate the need for vehicle specifications
		- Find an expression for price propagation between locations
		- Explain the model tying the theory together and its respective software
		- Discuss general conclusion, assumptions and further work
- # Background information at various levels
	- ## Basic
		- ### How rockets work
			- The main paradigm for space propulsion is chemical rockets: a fuel reacts chemically, is energized and accelerated, and propels the vehicle forward by flying out the back. The main fuel we will be considering is Hydrolox, or liquid hydrogen LH$_{2}$ and liquid oxygen LOx. This is because Hydrolox can be produced by electrolyzing (separating the hydrogen and oxygen) water, which exists in abundance on the Moon, making it the probable fuel used for cislunar transport systems.
			- Travelling in space is done mainly by consecutively changing your speed and orbit around some gravitational body. The ideal maneuver between two orbits or bodies is a [[Hohmann Transfer]] and requires several propellant burns, each changing the spacecraft's velocity by some $\Delta V$ [km/s]. The total $\Delta V$ for a trip between two locations also indicates how much energy (burned propellant) is required for transport.
			- The [[Mass Fraction]] $\eta$ of a single rocket maneuver is the ratio of the initial and final vehicle masses, i.e. after the required propellant has been burned.
			  
			  $$
			  \eta = m_{init} / (m_{init} - m_{prop})
			  $$
			  
			  $\eta$ also relates $v_e$ and $\Delta V$:
			  
			  $$
			  \eta = e^{\Delta V / v_e}
			  $$
			  
			  where $v_e$ is the fuel's [[Exhaust Velocity]] (which depends on the propellant used and is often represented by the engine's [[Specific Impulse]] $I_{sp} = v_e / g_0$). This means we can know $\eta$ solely from knowing our propellant-engine combination and the orbital trajectory being traversed.
			- The [[Maneuver Burn Time]] of a single propellant burn is described in the following equation
			  
			  $$
			  t_{burn}=\frac{m_{init} \cdot v_e}{T_{eng}}\left(1-e^{-\frac{\Delta V}{v_e}}\right)
			  $$
			  
			  where $T_{eng}$ is the engine's thrust. Since a mission often has several burns, the vehicle will be lighter at later burns due to the shed propellant mass.
			-
		- ### Some terms used
			- The **dry mass** of a rocket is its mass without any propellant or payload: all the systems, engines, structures, tanks, and cables.
		- ### Locations in cislunar space
			- Earth Surface
			- Low Earth Orbit
			- Earth Moon Lagrangian Points
			- Low Lunar Orbit
			- Lunar Surface
- # Cost drivers
	- When some cargo is shipped from A to B, transportation cost (as well as the profit margin) is the difference between the price paid before and after delivery. On Earth, shipping fees are a small fraction of the final price due to low propellant mass fractions $m_{prop}/m_{init}$ and extreme vehicle reusability. Ships and pickup trucks have a propellant mass fraction of 3%, cars 4%, cargo jets 40%, and rockets around 85% [[NASA](https://www.nasa.gov/mission_pages/station/expeditions/expedition30/tryanny.html)]. Container ships can be used intensely for 10 to 12 years [[Containercorp](https://www.containercorp.ca/how-many-times-can-a-shipping-container-be-used/)] and commercial airplanes for 25 to 30 years [[Quora](https://www.quora.com/How-many-years-can-passenger-airplanes-continue-to-fly-after-they-have-been-put-in-service?share=1)], but even the most reused rocket, a SpaceX Falcon 9, has only flown a total of 10 times [[Eclipse Aviation](https://www.eclipseaviation.com/how-many-times-has-spacex-reused-a-rocket/#6)] - the historical norm being just one expendable use. It is no wonder that space shipping makes up a large part of the final price of delivered payloads.
	- ## Components of transportation cost
		- ### Reusability-limited cost
			- Recouping the initial investment in the vehicle. The more uses your vehicle has, the more this recouping can be spread out: $C_{use} = C_{init}/ n_{cycles}$
			- For rockets, the lifetime of the engine is often the limit for the rest of the vehicle. Liquid propellant rocket engines have a limited amount of cycles (or restarts) $n_{cycles}$ and a total burn lifetime $t_{life}$, according to [this analysis](https://sci-hub.st/https://link.springer.com/article/10.1007/s12567-011-0006-x?shared-article-renderer):
			  
			  > [...] two major degradations limit the safe life operation of a rocket engine, the low-cycle fatigue damage related mainly to an engine start-up and shutdown operation, and the time-dependent damage driven by creep and material wear out related to the duration of the engine main stage operation.
			  >
			- These two limits, $n_{cycles}$ and $t_{life}$, are related by
			  
			  $$
			  n_{cycles} \cdot t_{burn, avg}= t_{life}
			  $$
			  
			  In case only one or the other value is known, they can be transformed into the other. Since trajectories have several burns $n_{burns}$ each, the **reusability-limited cost per trajectory** will be:
			  
			  $$
			  C_{use} = C_{init} \cdot \frac{n_{burns}}{n_{cycles}}  = C_{init} \cdot \frac{t_{burn}}{t_{life}}
			  $$
				- A further first level estimation from [this source](https://sci-hub.st/https://link.springer.com/article/10.1007/s12567-011-0006-x?shared-article-renderer) relates the average burn time to the Main Combustion Chamber (MCC) cycles to failure:
				  ![img](https://cdn.mathpix.com/snip/images/q1qR-S19h14D9UVmbX_L0F_vvW5lm33J2B3odPRcMQ4.original.fullsize.png)
				  
				  The equation graphed roughly equals $n_{uses} = 6024.0964 \cdot t_{burn, avg}^{-1}$, indicating that the average rocket in their analysis has a burn lifetime of around 6000 seconds. The authors suggest applying a safety margin factor of 3 to find the amount of safe cycles until failure - so their average rocket had a lifetime of around 2000 seconds.
		- ### Maintenance cost
			- Regular repair and maintenance. This cost has a fixed and variable part - one captures damage caused by simply using the vehicle, and the other depends on the conditions the vehicle must endure and the energetic distance of the journey: $C_{repair} = C_{repair, fixed} + C_{repair, var} \cdot \Delta V$
			- The final maintenance and repair cost is $C_{repair} = C_{init} \cdot (f_{repair,fixed} \cdot m_{dry} \cdot (1 + f_{repair,var} \cdot \Delta V))$
				- In order to make the expression easier to solve, we define these costs as a fraction of the initial vehicle cost:
				  
				  $$
				  C_{repair, fixed} = C_{init} \cdot r_{repair}
				  $$
				  
				  $$
				  \quad C_{repair, var} = C_{init} \cdot f_{repair, var}
				  $$
				- Furthermore, we define $r_{repair}$ as a function of the vehicle dry mass $m_{dry}$:
				  
				  $$
				  r_{repair} = f_{repair, fixed} \cdot m_{dry}
				  $$
					- The implication of this simplification is that the marginal damage suffered by the vehicle per $\Delta V$ travelled depends on the mass of the vehicle's systems and structures. The intuition behind this is that complexity and vulnerability scale with system mass, and since $\Delta V$ influences burn time more than travel time, it is also an expression of the operation of the engines (which are usually the main source of failure.)
		- ### Propellant cost
			- The cost of the fuel used: $C_{prop} = m_{prop} \cdot P_{prop}$ where $P_{prop}$ is the price paid for propellant at the origin.
	- ## Summary of cost equations
		- The three transportation costs are:
			- Reusability-limited cost:
			  $$
			  C_{use} = C_{init} \cdot \frac{t_{burn}}{t_{life}}  \quad  \text{or} \quad C_{init} \cdot \frac{n_{burns}}{n_{cycles}}
			  $$
			- Repair and maintenance cost:
			  $$
			  C_{repair} = C_{init} \cdot ((f_{repair, fixed} \cdot m_{dry} ) \cdot (1 + f_{repair, var} \cdot \Delta V ) )
			  $$
			- Propellant cost:
			  $$
			  C_{prop} =P_{prop} \cdot m_{prop}
			  $$
		- The expression found for all modelled costs is
		  $$
		  C_{total} = C_{init} \cdot ( \frac{t_{burn}}{t_{life}} + (f_{repair, fixed} \cdot m_{dry} ) \cdot (1 + f_{repair, var} \cdot \Delta V ) ) + P_{prop} \cdot m_{prop}
		  $$
- # Trajectory performance parameters
	- ## Rocket sizing 101
		- Spacecraft design often starts with a simple requirement: deliver a payload $m_{pay}$ from A to B. The chosen engine-propellant combo yields the Specific Impulse $I_{sp}$ and thus the Exhaust Velocity $v_e$. The orbital maneuvers required are found using the appropriate equations, yielding the $\Delta V$ for each burn.
		- The calculation of the initial mass required for the entire vehicle $m_{init}$ is performed backwards. Starting at the $n^{th}$ burn:
		  
		  $$
		  \eta_n = \frac{m_{dry} \cdot m_{pay} \cdot m_{prop,n}}{m_{dry} \cdot m_{pay}} = e^{ \frac{\Delta V_n}{v_e} }
		  $$
		  
		  $$
		  m_{prop, n} = (m_{dry} + m_{pay}) \cdot (\eta_n -1)
		  $$
		  
		  And with the propellant mass required for the final burn found, the previous burn's propellant mass can also be calculated.
		  
		  $$
		  m_{prop, n-1} = (m_{dry} + m_{pay} + m_{prop, n}) \cdot (\eta_{n-1} -1)
		  $$
		- If the vehicle is staged and thus has different $m_{dry}$ at different times, this must also be taken into account. This process is repeated until $m_{init}$ is reached.
	- ## Rocket sizing with mass ratios
		- If the actual mass of the payload is not known, it is possible to size a rocket, albeit using simplified coefficients instead.
			- The Payload Ratio $PR$ is defined as $PR = m_{pay} / (m_{dry} + m_{prop})$.
			- The Structural Coefficient $SC$ is defined as $SC = m_{dry} / (m_{dry} + m_{prop})$
			- With some algebra, it can be shown that $\eta = \frac{1 + PR}{SC + PR}$.
			- Clearly, a large payload ratio (more payload delivered), small structural coefficient (less dry mass needed), and large mass ratio are preferable.
		- Using mass ratios for vehicle sizing yields important performance parameters that remain true regardless of the absolute size of the vehicle.
		- Solving all the equations needed for concrete answers concerning payload masses delivered and costs can require in-depth knowledge of the vehicles being used. In order to make the model as generalizable as possible, simplifications and assumptions must be made in order to capture most of the result with less previous information.
		- As shown above, mass ratios express the performance of a vehicle for any mass it may have. In the previously derived expression for the total cost of a trajectory, there are three mass ratios which relate the dry, initial and propellant masses to that of the payload.
	- ## Finding the mass ratios
		- ### Adapting previous works
			- If you haven't yet visited Selenian Boondocks, I highly recommend the blog - a [recent article](https://selenianboondocks.com/2022/04/pf-derivation-split-dv-2/) by Kirk Sorensen saved me some work deriving the equations we're going to need. He considered the case of a vehicle that launches with a payload, drops it off somewhere, and continues on to another location (possibly the original starting point). I will spare you the derivation, which you can find in his article.
			- In Sorensen's model, he defines the dry mass of the craft (no propellant or payload) as depending on three terms: [initial-mass-sensitive](https://selenianboondocks.com/2010/02/calculating-gross-mass-sensitive-term/) ($\phi$), [propellant-mass-sensitive](https://selenianboondocks.com/2010/02/calculating-propellant-mass-sensitive-term/) ($\lambda$), and payload-mass-sensitive ($\epsilon$) mass terms. Typical values for in-space (vacuum) liquid hydrogen engines and propellant tanks are $\phi = 0.01$ and $\lambda = 0.03$.
			  
			  $$
			  m_{\text {dry }}=\phi \cdot m_{\text {initial }}+\lambda \cdot  m_{\text {prop }}+\epsilon \cdot m_{\text {payload }}
			  $$
			- Two further useful equations in his analysis are:
			  
			  $$
			  m_{init} = \eta_1 \cdot \eta_2 \cdot m_{dry} + \eta_1 \cdot m_{pay}
			  $$
			  
			  $$
			  m_{prop} = m_{init} \cdot (1 - \frac{1}{\eta_1 \cdot \eta_2}) - m_{pay} \cdot (1- \frac{1}{\eta_2})
			  $$
			- Using slick algebra, he arrives upon an expression relating the payload mass to the initial launch mass using only these constants:
			  
			  $$
			  \frac{m_{\text {init}}}{m_{\text {pay}}}=\frac{\eta_{1}\left(1+\epsilon \eta_{2}-\lambda\left(\eta_{2}-1\right)\right)}{1-(\phi+\lambda) \eta_{1} \eta_{2}+\lambda}
			  $$
			  
			  Here the $\eta$'s are the mass fractions $m_{init}/m_{final}$ for each maneuver of the mission (the second being the return trip sans the payload mass).
			- Rearranging $m_{init}/m_{pay}$ to find $m_{prop}/m_{pay}$:
			  
			  $$
			  \frac{m_{prop}}{m_{payload}} = \frac{\eta_{1}\left(1+\epsilon \eta_{2}-\lambda\left(\eta_{2}-1\right)\right)}{1-(\phi+\lambda) \eta_{1} \eta_{2}+\lambda} \cdot (1- \frac{1}{\eta_1 \eta_2}) - (1-\frac{1}{\eta_2})
			  $$
			- Finally, rearranging to express $m_{dry} / m_{pay}$ as a function of the mass-sensitive terms:
			  
			  $$
			  \frac{m_{dry}}{m_{pay}} = \phi \cdot \frac{m_{init}}{m_{pay}}+\lambda \cdot  \frac{m_{prop}}{m_{pay}}+\epsilon
			  $$
		- ### Three important considerations
			- In the case where the only cargo to be shipped is propellant itself, the payload is just more fuel. Since the payload-sensitive term ($\epsilon$) introduced by Sorensen represents the payload's unique contribution to the dry mass (e.g. as reinforcing structures to hold the cargo), a propellant-exclusive carrier which only has fuel tanks would have $\epsilon = 0$.
			- There are two scenarios which further simplify the mass ratio equations: if there is no return trip, then $\eta_2 =1$ since the craft's mass does not change. If the return trajectory has the same $\Delta V$ as the initial one, then $\eta_1 = \eta_2 = \eta$. This can only happen when there is no atmosphere to aerobrake in.
			- An underlying assumption of Sorensen's is a single-stage vehicle, mainly describing space-only tugs that don't need it. That means the mass ratio equations do not apply to multi-stage vehicles (anything launching from Earth).
	- ## Summary of mass ratio equations
		- ### Model Assumptions
			- The vehicle performs a maneuver $\eta_1$, drops off a payload $m_{pay}$, and performs a second maneuver $\eta_2$.
			- The dry mass is composed of contributions by
			  $$
			  m_{\text {dry }}=\phi \cdot m_{\text {init }}+\lambda \cdot  m_{\text {prop }}+\epsilon \cdot m_{\text {pay }}
			  $$
			- $\epsilon = 0$ for fuel tankers because the only payload being carried is propellant and thus all payload contributions to the dry mass are captured in $\lambda$.
		- ### One-way missions
			- $$
			  \frac{m_{prop}}{m_{payload}} = \frac{1- (\phi +\lambda) \eta +\lambda}{\eta} \cdot (1- \frac{1}{\eta})
			  $$
			- $$
			  \frac{m_{\text {init}}}{m_{\text {pay}}}=\frac{\eta_{1}}{1-(\phi+\lambda) \eta_{1} +\lambda}
			  $$
			- $$
			  \frac{m_{dry}}{m_{pay}} = \phi \cdot \frac{m_{init}}{m_{pay}}+\lambda \cdot  \frac{m_{prop}}{m_{pay}}
			  $$
		- ### Two-way missions
			- $$
			  \frac{m_{\text {init}}}{m_{\text {pay}}}=\frac{\eta_{1}\left(1-\lambda\left(\eta_{2}-1\right)\right)}{1-(\phi+\lambda) \eta_{1} \eta_{2}+\lambda}
			  $$
			- $$
			  \frac{m_{prop}}{m_{payload}} = \frac{1-(\phi+\lambda) \eta_{1} \eta_{2}+\lambda}{\eta_{1}\left(1-\lambda\left(\eta_{2}-1\right)\right)} \cdot (1- \frac{1}{\eta_1 \eta_2}) - (1-\frac{1}{\eta_2})
			  $$
			- $$
			  \frac{m_{dry}}{m_{pay}} = \phi \cdot \frac{m_{init}}{m_{pay}}+\lambda \cdot  \frac{m_{prop}}{m_{pay}}
			  $$
		- ### Sorensen's Mass-Sensitive Terms
			- Gross-Sensitive Mass Term
			  
			  $$
			  \phi=\frac{(T / W)_{\text {vehicle-initial }}\left(1+\left(f_{T S W}\right)(T / W)_{\text {vac }}\right)}{(T / W)_{\text {engine-initial }}}
			  $$
				- Term definitions:
					- $(T / W)_{\text{vehicle-init, engine-init, vac }}$ : Thrust to weight ratio (at the beginning of the mission) of the entire vehicle, only the engine, and the engine in vacuum
					- $f_{TSW}$ : Thrust-structure factor
			- Propellant-Sensitive Mass Term
			  
			  $$
			  \lambda=\frac{(M X R)\left(\frac{f_{O T}}{\rho_{o x}}\right)+\left(\frac{f_{F T}}{\rho_{f u e l}}\right)}{(1+M X R)\left(1-f_{\text {ullage }}\right)}
			  $$
				- Term definitions:
					- $MXR$ : Oxidizer to fuel mixture ratio $m_{ox}/m_{fuel}$
					- $f_{FT, OT}$ : Fuel tank, Oxidizer tank factors
					- $\rho_{fuel, ox}$ : Fuel, Oxidizer densities
					- $f_{ullage}$ : Ullage factor (extra tank volume)
- # Price propagation
	- The price of a good can be said to "propagate" when the good is physically moved in space and resold at a different price. The new price depends only on the parameters of the trajectory and the initial price, and thus the cost of moving cargo through the logistical network "propagates" alongside the cargo itself. Ideally, the effect of each trajectory on the price of goods travelling through it should be calculated as easily as possible and with as little information as possible.
	- If the target profit margin $f_{profit}$ for a trajectory from $A$ to $B$ is known beforehand, the required sale price $P_B$ can be found through:
	  
	  $$
	  P_{B} = f_{profit} \cdot (P_A + \frac{C_{total}}{m_{pay}})
	  $$
	- Focusing on and expanding $C_{total}/m_{pay}$:
	  
	  $$
	  \frac{C_{total}}{m_{pay}} =C_{init} \cdot \left[ \frac{v_e}{T_{eng}} \cdot \frac{(1-e^{\frac{ - \Delta V}{v_e}})}{t_{life}} \cdot \frac{m_{init}}{m_{pay}} + (f_{repair, fixed} \cdot (1 + f_{repair, var} \cdot \Delta V ) ) \cdot \frac{m_{dry}}{m_{pay}} \right] + P_{prop} \cdot \frac{m_{prop}}{m_{pay}}
	  $$
	- Plugging $C_{total} / m_{pay}$ back into the equation for $P_B$ yields a long expression that can be calculated with knowledge about the trajectory travelled $(\Delta V, n_{burns}, f_{profit})$, the node being departed from $(P_{prop}, P_{pay})$, the vehicle used to do so $(C_{init}, f_{repair, fix/var})$, and the engine-propellant combination used by the vehicle $(T_{eng}, v_e, t_{life})$.
	- ## Assumed behaviour for differing fuel prices
		- When the above expression for price propagation is implemented across many nodes in a location graph of cislunar space, it can model how the price of goods increases as they travel further and further from their origin. What happens, however, when two different fuel prices collide at a single location? Which fuel is used further downstream for each flow?
		- Another assumption is made in order to answer this question: at any given location, the cheapest local fuel will be used as the propellant for outgoing travel, while the original fuel of that flow is kept as the payload. That is, $P_{prop} = P_{best}$ and $P_{A} = P_{pay}$.
- # Software Implementation
	- This section will discuss how the theory above is actually transformed into a [modular tool](https://share.streamlit.io/ulfkemmsies/space_pricing_model/main) for creating transport graphs. It is highly recommended that you open the link in this block and try the functions out yourself.
	- The output of the model is a graph of **all the available refueling locations as nodes** with the **connecting edges** representing the **idealized travel between nodes** using a certain vehicle. Each edge (trajectory) incurs costs as described above and results in a higher payload price at the destination.
	  ![](https://cdn.mathpix.com/snip/images/eeiLR4QzddXPzL4CcL3VYdstMcwCMjvxb-Zbjld-NSc.original.fullsize.png)
	- ### Layout Clarifications
		- ![](https://cdn.mathpix.com/snip/images/kBJXmv5LRLByHQP7A8doAOn0dudue7RLC52sNwRprTc.original.fullsize.png)
		- The color of the arrows and nodes represents the propellant origin: blue is the Moon, green is the Earth. If there is only one color being shown (default is light blue), then only one fuel source's flow is being drawn.
		- A node being a certain color indicates that the fuel from the corresponding source is cheaper there - this allows us to quickly visualize the dominance of a certain fuel source.
		- The nodes and arrows can be hovered over for more complete information (all prices, mass ratios, etc.)
		  The "$k$" value shown on the arrows corresponds to $m_{prop}/m_{pay}$ in the above equations. Thicker arrows have a higher $k$ (as well as $\Delta V$).
		- The red arrow (sequence) is the best path from A to B in terms of propellant use. Calculating this in terms of total cost is in the pipeline.
	- ## Model Variables and Dependence
		- We shall call a specific output state the  "landscape" of cislunar space for shorthand. An important point is that (in the current version of the algorithm) the final landscape only depends on the **initial source fuel prices**, **vehicles used**, **trajectory options**, and **available refueling locations**.
		- **Vehicles** inherit the properties of an **Engine**, which inherits properties from its respective **Propellant**. All three can be defined separately and mixed and matched, but in order to get realistic outputs, we also need realistic assumptions. The engine's and propellant's qualities allow us to calculate $\phi$ and $\lambda$, while other values, like the cost coefficients $f_{repair}$, are defined at the vehicle level.
			- The 4 default vehicles, the *Space Tug*, *Lunar Lander*, *Aerobraker*, and *Lunar Aerobraker*, are defined by the design changes imposed by the ability to land on the Moon or aerobrake in LEO (or do both). The effect of lunar landing capability is a higher required thrust-to-weight ratio, and the effect of aerobraking is a 15% higher initial mass to accommodate the extra heatshields.
		- **Edges (trajectories)** have assigned **vehicles**,  $\Delta V$, **directionality** and **aerobraking** (both of which changes the mass ratio equations used above) variables. They also have set **profit margins** which all default to 10%.
		- **Nodes (locations)** at present only have **the presence of refueling and logistic infrastructure** expressed as a variable, but will in future have docking time/costs, resource capacity, etc. It would also be interesting to calculate prices at locations that do not have infrastructure to measure the impact of development surrounding it.
- # Thoughts, Assumptions, and Further Work
	- ## Thoughts and use cases
		- ### Thoughts
			- The idea behind being able to change the available variables quickly is to inspect the impact of certain changes on the entire system, e.g. how building fuel stations at _X_ location lowers system prices or improving _Y_ vehicle makes a certain path competitive. This impact is a potential metric for how worth it a specific improvement is as it relates to all of cislunar space's techno-industrial complex.
				- The necessity of such a metric arose out of the author's frustration with the misalignment between profitable activities in space (cubesats: sensing, navigation, telecom) and progress toward the fulfillment of the larger potential of space. As the profitability of large-scale cislunar development rises, and resulting re-alignment between the profit motive and industrial space expansion occurs, differentiating between financial gains and space-dev gains will be crucial for all types of players: investors, engineers, and policymakers among them.
			- Concerning some of the background rationale: the initial question that prompted all of this was "**When will the Moon become competitive?**", which, upon my realizing the high cost of space travel, prompted "**When and where will the Moon be competitive?**" This *where* element is what drove the graph-based representation of cislunar locations.
			- #### Why one-way rides rule
				- For the same vehicle and trajectory, how much more payload can you deliver (for the same propellant mass burned) by choosing a one-way mission and refueling upon arrival? Conversely, how much more propellant would a two-way trip need to deliver the same payload? Finding this advantage factor is easy:
				  $$
				  (\frac{m_{prop}}{m_{payload}})_2 / (\frac{m_{prop}}{m_{payload}})_1 -1 \rightarrow \frac{m_{payload_1}}{m_{payload_2}} -1 \quad\text{or}\quad \frac{m_{prop_2}}{m_{prop_1}} -1
				  $$
				- To visualize this, let's use typical values for in-space liquid oxygen engines and tanks: $\phi=0.01$, $\lambda=0.03$ and $v_e \approx 4400 m/s$. The x-axis is our $\Delta V$ in km/s. The purple and green line are the $m_{pay}/m_{prop}$ (the inverse of our definition!) ratios for one-way and two-way missions respectively, while the orange line is the advantage factor we defined earlier.
				- ![](https://cdn.mathpix.com/snip/images/GWLMo2_gTp7Nk0x3k2UepjCrIjR7IljeHoVz5EnOx4E.original.fullsize.png)
				- Even at the local minimum of the advantage factor, one-way trips still deliver nearly 100% more payload than their counterparts. This minimum is close to the $\Delta V$ between LEO and EML1, one of the longest energy-distances one would have to travel if there were refueling nodes at every important cislunar location. This advantage becomes incredible at longer and shorter distances, and specially the short trips are where refueling drastically changes the pricing of space transport. Six times your propellant in payload at 0.64 km/s, the gap between EML1 and LLO? Count me in!
				- In contrast, if one were to plot the $m_{payload}/m_{initial}$ advantage factor, it would quickly become apparent that it matters much less how much initial mass you start with: at around 4 km/s, a one-way trip would only get you around 15% more payload mass than a round trip. This difference only becomes significant after around 5 km/s, and even though the relative advantage factor takes off after 6 km/s, the absolute payload fraction is still tiny. This insensitivity of the payload fraction at low $\Delta V$'s means that filling up the tank completely doesn't make much of a difference for your delivery ratio, and so we will assume that all spacecraft are operating at full (tank) capacity upon departure.
				- ![](https://cdn.mathpix.com/snip/images/1ixKjofHeCpeV0zUAhYZX6ZDASecmzfPQoVXpWp9zqs.original.fullsize.png)
		- ### Preliminary results and use cases
			- A big disclaimer: since the vehicle coefficients and initial prices have not been tuned yet, the significance of any results shown here is tenuous at best (specially concerning specific numbers and prices). The examples are more of an illustration of the type of investigation one could conduct using the tool.
			- All of the examples below assume that Hydrolox costs 1 $/kg on Earth and 2000 \$/kg in LEO (according to the current launch prices on the market).
			- Iteratively finding the **best location for refueling** en route to the Moon:
				- For reference, the real-life price of lunar delivery straight from Earth ()at the time of writing) is 16MM$/kg by Astrobotic.
				- Going straight from LEO, our price multiplier is 6.4x.
				  ![image.png](../assets/image_1658456632274_0.png)
				- Adding Low Lunar Orbit as a refueling stop, the multiplier is now 5.65x. If all the other possible combinations of LEO + Moon + X node where tried (I did), it would quickly become apparent that this is the largest possible reduction in total lunar delivery cost from adding a single stop.
				  ![image.png](../assets/image_1658456701784_0.png)
				- Repeating the process for a second stop yields EML2 as the best next option, although the price reduction is truly minimal.
				  ![image.png](../assets/image_1658457095503_0.png)
				-
			- Finding the price at which **lunar ISRU breaks even**:
				- Assume fuel depots exist only in LLO (e.g. NASA's Lunar Gateway) as shown below.
				  ![image.png](../assets/image_1658457369981_0.png)
				- The breakeven point for lunar ISRU is when that station begins to buy from the Moon purely because of price.
				  ![image.png](../assets/image_1658457466585_0.png)
				- An even grander victory would be to beat Earth-based fuel in LEO (which will be the greatest market of all for a long time). Notice how this only happens once the cost of producing Hydrolox at the Moon falls to 500 $/kg - this number could be a goal for a lunar mining startup.
				  ![image.png](../assets/image_1658457590886_0.png)
				-
	- ## Assumptions
		- It cannot be overstressed that the quality of one's assumptions defines the quality of outputs, especially in this case, where there are so many assumptions to make.
		- The modelled starting point for Earth-based fuel is LEO. There are two reasons for this:
			- The use of staging invalidates Sorensen's mass ratio equations and thus the simplified framework for estimating vehicle performance.
			- The best known number in the space industry is dollars per $kg$ to LEO, and considering just how many guesstimates and assumptions are otherwise being made in this model, it is a good idea to anchor one in well-known values. This also implies that Earth-to-LEO transport will happen somewhat isolated from further cislunar shipping, as resetting Tsiolkovsky's equation early on yields massive payload delivery gains.
		- The $\Delta V$s between locations are collected from a variety of sources and are contained in the [[Cislunar Delta V Map]] that remains to be completed fully and one day turned into a self-contained function. When possible, aerobraking is used as an aeroassist for injecting into LEO and ballistic aerocapture when reentering directly to Earth. The difference is around 3 km/s for circularization in LEO.
	- ## Further Work
		- This model and app are nought but a starting point on a quest to model the economics of cislunar development. In no specific order, here is a bullet list of improvements and expansions to the theory and implementation that will bring us closer to reality and utility.
			- Realistically model the initial response to system changes (e.g. lunar lander making two-way trips to LLO until it becomes until one-way becomes possible through accumulated fuel)
			- Modeling real capacity of fuel and materials at nodes (they can be full)
			- Using different market shares of companies/vehicles to model percentage of all interactions happening like that (Markov Chains?)
			- Include Earth financial market behavior and how interest rates affect investment in cislunar infrastructure (by changing the discount rate and thus NPV)
			- Model how Earth launch prices affect the price of building nodes and ISRU over time
				- Consider how evolving Earth launch prices would influence the marginal mass needed for a lunar mining operation
			- Assume structural percentage of total mass for orbital structures and find price of construction taking differently sourced materials into account
			- Add the possibility of non-chemical launch systems, other fuels, other propulsion mechanisms (electric)
				- The massively cheapening effect of e.g. Orbital Rings, Launch Loops, Tethers, etc. must be expressed!
			- Add real time passing for price convergence calculation? Or perhaps consider how financial instruments like futures can affect these price changes?
			- The ability to change and persist variables on the variables page
	-