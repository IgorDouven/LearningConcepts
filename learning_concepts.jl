using PlotlyJS
using Distributed
using HypothesisTests
using EffectSizes
using GLM

addprocs(...) # set number of cores

@everywhere begin
    using CSV
    using Colors
    using DataFrames
    using Distributions
    using LinearAlgebra
    using Distances
    using Bootstrap
    using Flux
    using StatsBase
    import MLJ: partition
end

@everywhere coords = CSV.read(".../munsell_rgb.csv", DataFrame) |> Matrix # replace ... by path to directory in which the .csv file is stored

@everywhere function luv_convert(i)
	c = convert(Luv, RGB(coords[i, :]...))
	return c.l, c.u, c.v
end

@everywhere const luv_coords = [ luv_convert(i) for i in 1:size(coords, 1) ]

@everywhere prototypes = [[1.15, 0.16, -0.45], [98.89, 0.07, 0.51], [53.65, 0.54, -2.54], [26.45, -10.21, -104.25], [63.18, -59.74, 61.39], [94.55, 12.72, 79.97], [20.72, 27.86, 11.98], [45.67, 144.53, 21.34], [63.17, 112.57, 40.55], [28.98, 26.10, -67.19], [62.82, 96.05, -45.66]]


#########################################
## Visualization of CIELUV space, etc. ##
#########################################

labels = [ findmin([ evaluate(Euclidean(), prototypes[i], [luv_coords[j]...]) for i in 1:11 ])[2] for j in 1:size(luv_coords, 1) ]

proto_cols = [ convert(RGB, Luv(prototypes[i]...)) for i in 1:11 ]

cols = [ RGB(coords[i, :]...) for i in 1:size(coords, 1) ]

trace1 = scatter3d(x=first.(luv_coords),
                   y=getindex.(luv_coords, 2),
                   z=last.(luv_coords),
				   mode="markers", 
                   marker_size=1, # marker_size=1, if shown together with prototypes
                   marker_color=cols)

trace2 = scatter3d(x=first.(prototypes),
                   y=getindex.(prototypes, 2),
                   z=last.(prototypes),
				   mode="markers", 
                   marker_size=5.5,
                   marker_color=proto_cols)

layout = Layout(font_family = "Cabin", width=1100, height=800, margin=attr(l=0, r=0, t=0, b=0), showlegend = false,
		        scene=attr(xaxis=attr(title=attr(text="<i>L</i><sup>*</sup>", 
			               font_size=14)), yaxis=attr(title=attr(text="<i>u</i><sup>*</sup>",
			               font_size=14)), zaxis=attr(title=attr(text="<i>v</i><sup>*</sup>", font_size=14)), aspectratio=attr(x=.8, y=1, z=1.1),
			               aspectmode = "manual",
			               #camera=attr(up=attr(x=1,y=0,z=0), center=attr(x=0,y=0,z=0), eye=attr(x=1.25,y=1.5,z=1.5))))
			               camera=attr(up=attr(x=65,y=-140,z=110), center=attr(x=0,y=0,z=0), eye=attr(x=1.25,y=1.25,z=-1.5))))

p = PlotlyJS.plot([trace1, trace2], layout)

savefig(p, "proto_2_light.pdf"; scale=1)

tr(i) = scatter3d(x=first.(luv_coords[labels .== i]),
                  y=getindex.(luv_coords[labels .== i], 2),
                  z=last.(luv_coords[labels .== i]),
			      mode="markers", 
                  marker_size=7.5,
                  marker_color=cols[labels .== i])

# for plotting separate colors, or several of them
PlotlyJS.plot([trace1, tr(5), tr(10)], layout)


#####################
## Learning colors ##
#####################

@everywhere prototypes = convert(Vector{Vector{Float32}}, prototypes)

@everywhere random_constellations = [ convert(Vector{Vector{Float32}}, hcat(collect.(luv_coords[sample(1:1625, 11; replace=false)]))[:]) for _ in 1:1000 ]

@everywhere function _compare_constellations(c::Vector{Vector{Float32}}, epochs::Int)
    train, test = partition(luv_coords, .8; shuffle=true)
    labs_train = [ findmin([ evaluate(Euclidean(), c[i], [train[j]...]) for i in 1:11 ])[2] for j in 1:size(train, 1) ]
    labs_test = [ findmin([ evaluate(Euclidean(), c[i], [test[j]...]) for i in 1:11 ])[2] for j in 1:size(test, 1) ]
    hc_train = hcat(train, labs_train)
    hc_test = hcat(test, labs_test)
    x_train = collect.(hc_train[:, 1])
    x_train = convert(Vector{Vector{Float32}}, x_train)
    df_train = reduce(hcat, x_train)
    y_train = Flux.onehotbatch(hc_train[:, 2], 1:11)
    x_test = collect.(hc_test[:, 1])
    x_test = convert(Vector{Vector{Float32}}, x_test)
    df_test = reduce(hcat, x_test)
    l1 = Dense(3, 9, Flux.relu)
    l2 = Dense(9, 11)
    Flux_nn = Flux.Chain(l1, l2)
    loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
    ps = Flux.params(Flux_nn)
    nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
    cc_train = Float32[] # classification correctness training set
    cc_test = Float32[] # classification correctness test set
    for epoch in 1:epochs
        Flux.train!(loss, ps, nndata, Flux.ADAM())
        ŷ_train = Flux.onecold(Flux_nn(df_train), 1:11)
        push!(cc_train, sum(labs_train .== ŷ_train) / length(train))
        ŷ_test = Flux.onecold(Flux_nn(df_test), 1:11)
        push!(cc_test, sum(labs_test .== ŷ_test) / length(test))
    end
    return cc_train, cc_test
end

@everywhere function compare_constellations(rc::Vector{Vector{Float32}}, epochs::Int)
    train, test = partition(luv_coords, .8; shuffle=true)
    proto = _compare_constellations(prototypes, epochs)
    rnd = _compare_constellations(rc, epochs)
    return proto, rnd
end

cc_res = pmap(i->compare_constellations(random_constellations[i], 1000), 1:length(random_constellations))

bs(x; n=1000) = bootstrap(mean, x, BasicSampling(n))

ptr = reduce(hcat, first.(first.(cc_res)))
ptr_bs = [ Bootstrap.confint(bs(ptr[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
ptst = reduce(hcat, last.(first.(cc_res)))
ptst_bs = [ Bootstrap.confint(bs(ptst[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

rtr = reduce(hcat, first.(last.(cc_res)))
rtr_bs = [ Bootstrap.confint(bs(rtr[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
rtst = reduce(hcat, last.(last.(cc_res)))
rtst_bs = [ Bootstrap.confint(bs(rtst[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

ttest_train(j) = pvalue(EqualVarianceTTest(reduce(hcat, first.(first.(cc_res)))[j, :], reduce(hcat, first.(last.(cc_res)))[j, :]))
ttest_test(j) = pvalue(EqualVarianceTTest(reduce(hcat, last.(first.(cc_res)))[j, :], reduce(hcat, last.(last.(cc_res)))[j, :]))

[ ttest_train(i) for i in 1:1000 ]
[ ttest_test(i) for i in 1:1000 ]

cohd_train(j) = effectsize(CohenD(Float64.(reduce(hcat, first.(first.(cc_res)))[j, :]), Float64.(reduce(hcat, first.(last.(cc_res)))[j, :]), quantile=.95))
cohd_test(j) = effectsize(CohenD(Float64.(reduce(hcat, last.(first.(cc_res)))[j, :]), Float64.(reduce(hcat, last.(last.(cc_res)))[j, :]), quantile=.95))

cohd_tr = [ cohd_train(i) for i in 50:1000 ] # small: d = 0.2; medium: d = 0.5; large: d = 0.8
cohd_tst = [ cohd_test(i) for i in 50:1000 ]

epochs = 50:1000
df_train = DataFrame(epochs=epochs, proto_res=first.(ptr_bs)[epochs], ymin1=getindex.(ptr_bs, 2)[epochs], ymax1=last.(ptr_bs)[epochs], rnd_res=first.(rtr_bs)[epochs], ymin2=getindex.(rtr_bs, 2)[epochs], ymax2=last.(rtr_bs)[epochs])
df_test = DataFrame(epochs=epochs, proto_res=first.(ptst_bs)[epochs], ymin1=getindex.(ptst_bs, 2)[epochs], ymax1=last.(ptst_bs)[epochs], rnd_res=first.(rtst_bs)[epochs], ymin2=getindex.(rtst_bs, 2)[epochs], ymax2=last.(rtst_bs)[epochs])

traces1 = [
    scatter(x = 50:1000, y = cohd_tr, name = "Cohen's <i>d</i>", yaxis = "y2", line_width=1.5, line_color="rgba(48, 48, 48, 0.4)"),
    
    scatter(x=df_train.epochs, y=df_train.ymin2, line_width=0, showlegend=false),
    scatter(x=df_train.epochs, y=df_train.ymax2, fill="tonexty", fillcolor="rgba(238, 44, 44, .4)", line_width=0, showlegend=false),
    scatter(x=df_train.epochs, y=df_train.rnd_res, mode="lines", line_width=2, line_color="rgb(238, 44, 44)", name="Random"),
        
    scatter(x=df_train.epochs, y=df_train.ymin1, line_width=0, showlegend=false),
    scatter(x=df_train.epochs, y=df_train.ymax1, fill="tonexty", fillcolor="rgba(25, 25, 112, .4)", line_width=0, showlegend=false),
    scatter(x=df_train.epochs, y=df_train.proto_res, mode="lines", line_width=2, line_color="rgb(25, 25, 112)", name="Actual")
]    

layout1 = Layout(
  font_family = "Cabin",
  legend = attr(x = 1.075, y= 1.001,),
  xaxis_range = [0, 1000],
  yaxis_range = [.835, .975],
  title = "Training accuracy",
  xaxis_title = "Epoch",
  yaxis_title = "Accuracy",
  yaxis2 = attr(
    range = [.4, 1.3],
    title = "<i>d</i>",
    overlaying = "y",
    side = "right",
    showgrid = false
  )
);

pp1 = PlotlyJS.Plot(traces1, layout1)

savefig(pp1, "training.pdf"; width=590, height=310, scale=1)

traces2 = [
    scatter(x = 50:1000, y = cohd_tst, name = "Cohen's <i>d</i>", yaxis = "y2", line_width=1.5, line_color="rgba(48, 48, 48, 0.4)"),
    
    scatter(x=df_test.epochs, y=df_test.ymin2, line_width=0, showlegend=false),
    scatter(x=df_test.epochs, y=df_test.ymax2, fill="tonexty", fillcolor="rgba(238, 44, 44, .4)", line_width=0, showlegend=false),
    scatter(x=df_test.epochs, y=df_test.rnd_res, mode="lines", line_width=2, line_color="rgb(238, 44, 44)", name="Random"),
        
    scatter(x=df_test.epochs, y=df_test.ymin1, line_width=0, showlegend=false),
    scatter(x=df_test.epochs, y=df_test.ymax1, fill="tonexty", fillcolor="rgba(25, 25, 112, .4)", line_width=0, showlegend=false),
    scatter(x=df_test.epochs, y=df_test.proto_res, mode="lines", line_width=2, line_color="rgb(25, 25, 112)", name="Actual")
]    

layout2 = Layout(
  font_family = "Cabin",
  legend = attr(x = 1.075, y= 1.001,),
  xaxis_range = [0, 1000],
  yaxis_range = [.835, .975],
  title = "Validation accuracy",
  xaxis_title = "Epoch",
  yaxis_title = "Accuracy",
  yaxis2 = attr(
    range = [.4, 1.3],
    title = "<i>d</i>",
    overlaying = "y",
    side = "right",
    showgrid = false
  )
);

pp2 = PlotlyJS.Plot(traces2, layout2)

savefig(pp2, "validation.pdf"; width=590, height=310, scale=1)

# Might learnability be explained by contrastiveness and/or representativeness, and thus not add anything, or add little, on its own?
# To check, we run a regression analysis.

function calc(ar::Vector{Vector{Float32}}) # calculates contrastiveness and representativeness of a constellation
    c = sum(LowerTriangular(pairwise(Euclidean(), reduce(hcat, ar))))
    find_color(x) = [ findmin([ evaluate(Euclidean(), x[i], [luv_coords[j]...]) for i in 1:11 ])[2] for j in 1:size(luv_coords, 1) ]
    fc = find_color(ar)
    dfg = DataFrame(first=first.(luv_coords), second=getindex.(luv_coords, 2), third=last.(luv_coords), group=fc)
    gd = groupby(dfg, :group)
    cd = combine(gd, valuecols(gd) .=> mean)
    centers = convert(Matrix{Float32}, Matrix(cd[:, 2:4]))
    r = 0.0
    for i in 1:11
        r += euclidean(centers[i, :], reduce(hcat, ar)'[i, :])
    end
    return [c, r]
end

cr = calc.(random_constellations)

function reg_mod(i::Int)
    df = DataFrame(DV=Float64.(rtr[i, :]), IV1=first.(cr), IV2=last.(cr))
    mod = lm(@formula(DV ~ IV1 * IV2), df)
    return coeftable(mod).cols[4][2:4]
end

reg_res = [ reg_mod(i) for i in 1:1000 ]
sum(collect(Iterators.flatten(reg_res)) .< .05)

