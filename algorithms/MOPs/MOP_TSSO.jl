import XLSX
using JuMP, Gurobi
using ProgressMeter
using DataFrames
using CSV
using NPZ

include(joinpath(@__DIR__,"mop_utils.jl"))

sys_file_name = "123Bus_Data.xlsx"
scenario_file_name = "123Bus_50Scenario_Data.xlsx" # For comuptional efficiency, we only use the clustered 50 scenarios
args = read_data(sys_file_name, scenario_file_name)


NT, NS, N_TL, N_NL, N_vulne_branch, Branch_start, N_Branch, TL_Mask, NL_Mask, N_Bus, pIn, pInn, N_DG, DG_Mask, R_Branch0, X_Branch0, Big_M_V, V0,
        V_min, V_max, Pd, Qd, SZ_Branch0, S_Branch0, P_DG_min0, P_DG_max, Q_DG_min0, Q_DG_max, BigM_SC, BSDG_Mask,Mask_NoLoad_Bus,
        Big_M_FF, atk, Sb = args

model = Model()

# Stage 1:
#---------MGF------------------
# pu.
@variable(model, FF[1:N_Branch,1:NT]) # commodity flow at line ij
# binary
@variable(model, b[1:N_Branch,1:NT], Bin) # switch state of line ij
@variable(model, X_EN[1:N_Bus,1:NT], Bin)
@variable(model, X_BS[1:N_Bus,1:NT], Bin) # Black start ablility
@variable(model, z_bs1[1:N_Bus,1:NT], Bin) # 
@variable(model, z_bs2[1:N_Bus,1:NT], Bin)
@variable(model, X_tieline[1:N_TL,1:NT], Bin) # line final state.
@variable(model, X_line[1:N_NL,1:NT], Bin)
#--------PDN--------------------


# Stage-2
#---------MGF------------------
# pu.
@variable(model, FF_AF[1:N_Branch,1:NT,1:NS]) # commodity flow at line ij
# binary
@variable(model, b_AF[1:N_Branch,1:NT,1:NS], Bin) # switch state of line ij
@variable(model, X_EN_AF[1:N_Bus,1:NT,1:NS], Bin)
@variable(model, z_bs[1:N_Bus,1:NT,1:NS], Bin) # for  X_EN_AF * X_BS
@variable(model, X_rec[1:N_Bus,1:NT,1:NS], Bin) # load pick up
@variable(model, X_line_AF[1:N_NL,1:NT,1:NS], Bin)
# --------PDN-------------------
@variable(model, PF[1:N_Branch,1:NT,1:NS]) # active power flow at line ij
@variable(model, QF[1:N_Branch,1:NT,1:NS]) # reactive power flow at line ij
@variable(model, V[1:N_Bus,1:NT,1:NS]) 
@variable(model, P_dg[1:N_DG,1:NT,1:NS]) #
@variable(model, Q_dg[1:N_DG,1:NT,1:NS])
@variable(model, P_dg_curtailment[1:N_DG,1:NT,1:NS])
@variable(model, Q_dg_curtailment[1:N_DG,1:NT,1:NS])
@variable(model, Pd_rec[1:N_Bus,1:NT,1:NS])
@variable(model, Qd_rec[1:N_Bus,1:NT,1:NS])
@variable(model, Pd_curtailment[1:N_Bus,1:NT,1:NS])
@variable(model, Qd_curtailment[1:N_Bus,1:NT,1:NS])
@variable(model, PL_loss[1:N_Branch,1:NT,1:NS])


#cons

# Stage-1
@constraint(model, X_BS .== repeat(sum(BSDG_Mask,dims=2),1,NT))
@constraint(model, pIn * FF .+ X_EN .<= Big_M_FF .* (1 .- z_bs1))
@constraint(model, pIn * FF .+ X_EN .>= -Big_M_FF .* (1 .- z_bs1))
@constraint(model, z_bs1 .- 1 .<= X_BS)
@constraint(model, X_BS .<= 1 .- z_bs1)
@constraint(model, pIn * FF .>= -Big_M_FF .* (1 .- z_bs2))
@constraint(model, z_bs2 .- 1 .<= X_BS .- 1)
@constraint(model, X_BS .- 1 .<= 1 .- z_bs2)
@constraint(model, X_EN .- X_BS .>= -Big_M_FF .* (1 .- z_bs2))
@constraint(model, X_EN .- X_BS .<= Big_M_FF .* (1 .- z_bs2))
@constraint(model, z_bs1 .+ z_bs2 .== 1)
@constraint(model, -Big_M_FF .* b .<= FF)
@constraint(model, FF .<= Big_M_FF .* b)
@constraint(model, sum(b,dims=1) .== N_Bus .- sum(X_BS,dims=1) .- sum(1 .- X_EN,dims=1))
@constraint(model, X_line .== NL_Mask' * b)

# Stage-2
# --Constraints--
# Power flow bus PQ blance
for s in 1:NS
    @constraint(model, pIn * PF[:,:,s] .== DG_Mask * P_dg[:,:,s] .- Pd_rec[:,:,s])
    @constraint(model, pIn * QF[:,:,s] .== DG_Mask * Q_dg[:,:,s] .- Qd_rec[:,:,s])

    # Power flow voltage
    @constraint(model, V0 .* (pIn' * V[:,:,s]) .- repeat(R_Branch0,1,NT) .* PF[:,:,s] 
        .- repeat(X_Branch0,1,NT) .* QF[:,:,s] .<= Big_M_V .* (1 .- b_AF[:,:,s]))
    @constraint(model, V0 .* (pIn' * V[:,:,s]) .- repeat(R_Branch0,1,NT) .* PF[:,:,s] 
        .- repeat(X_Branch0,1,NT) .* QF[:,:,s] .>= -Big_M_V .* (1 .- b_AF[:,:,s]))
    @constraint(model, V0 .* X_BS .+ X_EN_AF[:,:,s] .* V_min .- z_bs[:,:,s] .* V_min .<= V[:,:,s])
    @constraint(model, V[:,:,s] .<= X_BS .* V0 .+ X_EN_AF[:,:,s] .* V_max .- z_bs[:,:,s] .* V_max)
    @constraint(model, z_bs[:,:,s] .<= X_BS)
    @constraint(model, z_bs[:,:,s] .<= X_EN_AF[:,:,s])
    @constraint(model, z_bs[:,:,s] .>= X_BS .+ X_EN_AF[:,:,s] .- 1)
end

# 3. % 3. Load Curtailments
@constraint(model, X_rec .<= X_EN_AF)
@constraint(model, Pd_rec .>= 0)
@constraint(model, Qd_rec .>= 0)
@constraint(model, Pd_curtailment .== Pd - Pd_rec)
@constraint(model, Qd_curtailment .== Qd - Qd_rec)
@constraint(model, Pd_rec .<= X_rec .* Pd)
@constraint(model, Qd_rec .<= X_rec .* Qd)

for s in 1:NS
    @constraint(model, Mask_NoLoad_Bus' * X_rec[:,:,s] .== 0)
end

# Power flow thermal limits
@constraint(model, PF .>= -repeat(S_Branch0,1,NT,NS) .* b_AF)
@constraint(model, PF .<= repeat(S_Branch0,1,NT,NS) .* b_AF)
@constraint(model, QF .>= -repeat(S_Branch0,1,NT,NS) .* b_AF)
@constraint(model, QF .<= repeat(S_Branch0,1,NT,NS) .* b_AF)

# DG
for s in 1:NS
    @constraint(model, P_dg[:,:,s] .>= (DG_Mask'*X_EN_AF[:,:,s]) .* repeat(P_DG_min0,1,NT)) 
    @constraint(model, P_dg[:,:,s] .<= (DG_Mask'*X_EN_AF[:,:,s]) .* P_DG_max[:,:,s])
    @constraint(model, Q_dg[:,:,s] .>= (DG_Mask'*X_EN_AF[:,:,s]) .* repeat(Q_DG_min0,1,NT))
    @constraint(model, Q_dg[:,:,s] .<= (DG_Mask'*X_EN_AF[:,:,s]) .* Q_DG_max[:,:,s])
end
@constraint(model, P_dg_curtailment .== P_DG_max - P_dg)
@constraint(model, Q_dg_curtailment .== Q_DG_max - Q_dg)

 # mgf after attack
for s in 1:NS
    @constraint(model, pIn * FF_AF[:,:,s] .+ X_EN_AF[:,:,s] .<= Big_M_FF .* (1 .- z_bs1))
    @constraint(model, pIn * FF_AF[:,:,s] .+ X_EN_AF[:,:,s] .>= -Big_M_FF .* (1 .- z_bs1))
    @constraint(model, pIn * FF_AF[:,:,s] .>= -Big_M_FF .* (1 .- z_bs2))
    @constraint(model, sum(b_AF[:,:,s],dims=1) .== N_Bus .- sum(X_BS,dims=1) .- sum(1 .- X_EN_AF[:,:,s],dims=1))
end

@constraint(model, X_EN_AF .- repeat(X_BS,1,1,NS) .>= -Big_M_FF .* (1 .- repeat(z_bs2,1,1,NS)))
@constraint(model, X_EN_AF .- repeat(X_BS,1,1,NS) .<= Big_M_FF .* (1 .- repeat(z_bs2,1,1,NS)))
@constraint(model, -Big_M_FF .* b_AF .<= FF_AF)
@constraint(model, FF_AF .<= Big_M_FF .* b_AF)
@constraint(model, b_AF .<= atk ) #NOTE: Requiring fixing a

# Obj
@objective(model, Min, (sum(Pd_curtailment[:])+ sum(Qd_curtailment[:]) + sum(P_dg_curtailment[:])
                        + sum(Q_dg_curtailment[:]) - 0.1*sum(X_line_AF[:]) )/NS  - 0.1*sum(X_line[:]))
# @objective(model, Min, 0)
set_optimizer(model, Gurobi.Optimizer)
set_optimizer_attributes(model, "MIPGap"=> 1.0e-2)

time_elapsed = @elapsed begin
    optimize!(model)
end
println("Time elapsed: ", time_elapsed, " seconds")

res_b = value.(b)
# save the results
df = DataFrame(res_b,:auto)
CSV.write("res_b50.csv", df)