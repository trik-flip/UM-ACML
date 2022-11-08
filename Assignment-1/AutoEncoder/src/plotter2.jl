using Plots
default(legend=false)
x = y = range(-5, 5, length=40)
zs = zeros(0, 40)
n = 1000

@gif for i in range(0, stop=10 * 2π, length=n)
    f(x, y) = sin(x + 10sin(i)) + cos(y)

    l = @layout[a{0.7w} b; c{0.2h}]
    p = plot(x, y, f, st=[:surface, :contourf], layout=l)

    plot!(p[1], camera=(10 * (1 + cos(i)), 40))

    fixed_x = zeros(40)
    z = map(f, fixed_x, y)
    plot!(p[1], fixed_x, y, z, line=(:black, 5, 0.2))
    vline!(p[2], [0], line=(:black, 5))

    global zs = vcat(zs, z')
    plot!(p[3], zs, alpha=0.2, palette=cgrad(:blues).colors)

end