"""
cable_monitor
=============

Robuuste, monitoring-only catenary-monitor met anomaliedetectie voor een
maritieme kabel die via een geleiderrol de kade op loopt.

Ontwerpprincipes
----------------
1. MONITORING ONLY. Deze software bedient GEEN machines. Hij meet, beoordeelt
   en visualiseert; de operator grijpt in. Er zit bewust geen actuator/noodstop
   in (geen SIL/PL veiligheidsfunctie).

2. HEALTH-AWARE. Elke meting krijgt een betrouwbaarheids-/zicht-score. Bij
   regen, sneeuw, beslagen lens of duisternis meldt de monitor eerlijk
   "zicht onbetrouwbaar" in plaats van valse anomalieen te produceren.

3. SOURCE-AGNOSTIC. De beeldbron zit achter een interface (`FrameSource`),
   zodat een DepthAI-camera, een simulator, of later een IR/nacht-bron
   inwisselbaar zijn zonder de rest te herschrijven.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
