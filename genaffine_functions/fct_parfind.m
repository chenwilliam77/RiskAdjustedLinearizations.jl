function value = fct_parfind( MODEL , string)

value = MODEL.calibration.params(MODEL.parameters.params==string);