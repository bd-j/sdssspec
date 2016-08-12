plot(obsdat['wavelength'], obsdat['spectrum'])
s, p, _ = model.mean_model(model.theta, obs=obsdat, sps=sps)
plot(obsdat['wavelength'], s)
