function fout = BFieldNoise_ONEPHASE( dur_dark, noisex, noisey, noisez, magx, magy, magz, zpol, B0fac, g, randseed )
  save_values = []
  rng(randseed);
  close all;
  
  dt = 1e-3;
  t_n_range = linspace( 0, dur_dark, dur_dark/dt );
  
  B0 = 5e-3*B0fac
  
  Bxn = noisex * normrnd(0, 10^(-magx), length(t_n_range), 1);
  Byn = noisey * normrnd(0, 10^(-magy), length(t_n_range), 1);
  Bzn = noisez * normrnd(0, 10^(-magz), length(t_n_range), 1);
  
  Bv0 = [ Bxn Byn B0+Bzn ];
  
  sol_values = ode5(@(t, Pv) diff_eq(t, Pv, Bv0, t_n_range, g), t_n_range, [sqrt(1-zpol*zpol) 0. zpol] );
  save_values = [ sol_values(:, 1) sol_values(:, 2) sol_values(:, 3) ];
  filename = sprintf('Simulations/noiseallsim_saveall_dark-%d_x-%d_y-%d_z-%d_magx-%d_magy-%d_magz-%d_zpol-%d_B0fac-%d_g-%d_sol.txt', dur_dark, noisex, noisey, noisez, magx, magy, magz, zpol, B0fac, g )
  dlmwrite(filename, save_values, 'delimiter', ' ', 'precision', 10);
end

function dPvdt = diff_eq( t, Pv, Bv0, t_range, g )
  [t_c t_ind] = min(abs(t - t_range));
  Bv = [ Bv0(t_ind, 1) Bv0(t_ind, 2) Bv0(t_ind, 3) ];

  dPvdt = g * cross( Pv(1:3), Bv )';
end

