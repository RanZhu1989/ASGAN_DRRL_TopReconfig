import XLSX
using JuMP, Gurobi
using ProgressMeter
using DataFrames
using CSV
using NPZ
include(joinpath(@__DIR__,"mop_utils.jl"))


function Deterministic_OneStep(args, ind_S, ind_T)
    NT, NS, N_TL, N_NL, N_vulne_branch, Branch_start, N_Branch, TL_Mask, NL_Mask, N_Bus, pIn, pInn, N_DG, DG_Mask, R_Branch0, X_Branch0, Big_M_V, V0,
            V_min, V_max, Pd, Qd, SZ_Branch0, S_Branch0, P_DG_min0, P_DG_max, Q_DG_min0, Q_DG_max, BigM_SC, BSDG_Mask,Mask_NoLoad_Bus,
            Big_M_FF, atk, Sb = args

    model = Model()


    @variable(model, PF[1:N_Branch]) # active power flow at line ij
    @variable(model, QF[1:N_Branch]) # reactive power flow at line ij
    @variable(model, SV[1:N_Bus]) # voltage^2 at bus j
    @variable(model, SI[1:N_Branch])
    @variable(model, P_dg[1:N_DG]) # 
    @variable(model, Q_dg[1:N_DG])
    @variable(model, P_dg_curtailment[1:N_DG])
    @variable(model, Q_dg_curtailment[1:N_DG])
    @variable(model, Pd_rec[1:N_Bus])
    @variable(model, Qd_rec[1:N_Bus])
    @variable(model, Pd_curtailment[1:N_Bus])
    @variable(model, Qd_curtailment[1:N_Bus])
    @variable(model, PL_loss[1:N_Branch]) # line loss
    @variable(model, FF[1:N_Branch]) # commodity flow at line ij

    @variable(model, X_rec[1:N_Bus], Bin) # load pick up
    @variable(model, X_EN[1:N_Bus], Bin)
    @variable(model, X_tieline[1:N_TL], Bin) # line final state. 
    @variable(model, X_line[1:N_NL], Bin)

    @variable(model, z_bs[1:N_Bus], Bin) # for bilinear term
    @variable(model, b[1:N_Branch], Bin) # switch state of line ij
    @variable(model, X_BS[1:N_Bus], Bin) # black start ablility
    @variable(model, z_bs1[1:N_Bus], Bin) # for X_EN * X_BS
    @variable(model, z_bs2[1:N_Bus], Bin)

    # ------------------powerflow--------------------
    # 1. Bus PQ Blance: S_jk - S_ij = S_inj
    @constraint(model, pIn * PF - pInn * (R_Branch0.*SI) .== DG_Mask * P_dg .- Pd_rec)  
    @constraint(model, pIn * QF - pInn * (S_Branch0.*SI) .== DG_Mask * Q_dg .- Qd_rec)

    # 2. Voltage : v_i-v_j=2Re(z_ij·S_ij*)+(r.^2+x.^2).*l_ij=2(r·P_ij,i+x·Q_ij,i)+(r.^2+x.^2).*l_ij
    @constraint(model, pIn' * SV .- 2 .* R_Branch0 .* PF .- 2 .* X_Branch0 .* QF + SZ_Branch0 .* SI.<= Big_M_V .* (1 .- b))
    @constraint(model, pIn' * SV .- 2 .* R_Branch0 .* PF .- 2 .* X_Branch0 .* QF + SZ_Branch0 .* SI .>= -Big_M_V .* (1 .- b))
    @constraint(model, V0^2 .* X_BS .+ X_EN .* V_min^2 .- z_bs .* V_min^2 .<= SV)
    @constraint(model, SV .<= X_BS .* V0^2 .+ X_EN .* V_max^2 .- z_bs .* V_max^2)
    @constraint(model, z_bs .<= X_BS)
    @constraint(model, z_bs .<= X_EN)
    @constraint(model, z_bs .>= X_BS .+ X_EN .- 1)
    for i in 1:N_Branch
        j = Branch_start[i]  # sl is the node index of starting node of line l
        # Add a rotated second order cone constraint
        @constraint(model, [0.5*SV[j]; SI[i]; [PF[i]; QF[i]]] in RotatedSecondOrderCone())
    end

    # 3. Load Curtailments
    @constraint(model, X_rec .<= X_EN)
    @constraint(model, Mask_NoLoad_Bus' * X_rec .== 0)
    @constraint(model, Pd_rec .>= 0)
    @constraint(model, Qd_rec .>= 0)
    @constraint(model, Pd_curtailment .== Pd[:,ind_T,ind_S] - Pd_rec)
    @constraint(model, Qd_curtailment .== Qd[:,ind_T,ind_S] - Qd_rec)
    @constraint(model, Pd_rec .<= X_rec .* Pd[:,ind_T,ind_S])
    @constraint(model, Qd_rec .<= X_rec .* Qd[:,ind_T,ind_S])

    # % 4. line
    @constraint(model, SI .>= -S_Branch0 .* b)
    @constraint(model, SI .<= S_Branch0 .* b)
    @constraint(model, SI .>= -S_Branch0 .* b)
    @constraint(model, SI .<= S_Branch0 .* b)
    @constraint(model, PL_loss .== R_Branch0 .* SI)

    # ------------DG ---------value.Pd_rec-------
    @constraint(model, P_dg .>= (DG_Mask'*X_EN) .* P_DG_min0) 
    @constraint(model, P_dg .<= (DG_Mask'*X_EN) .* P_DG_max[:,ind_T,ind_S])
    @constraint(model, Q_dg .>= (DG_Mask'*X_EN) .* Q_DG_min0)
    @constraint(model, Q_dg .<= (DG_Mask'*X_EN) .* Q_DG_max[:,ind_T,ind_S])
    @constraint(model, P_dg_curtailment .== P_DG_max[:,ind_T,ind_S] - P_dg)
    @constraint(model, Q_dg_curtailment .== Q_DG_max[:,ind_T,ind_S] - Q_dg)

    # Commodity flow based topology logic
    @constraint(model, X_BS .== sum(BSDG_Mask,dims=2))
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
    @constraint(model, b .<= atk[:,ind_T, ind_S] ) #NOTE: Requiring fixing a
    @constraint(model, sum(b) == N_Bus - sum(X_BS) - sum(1 .- X_EN))
    @constraint(model, X_line .== NL_Mask' * b)

    # Obj
    @objective(model, Min, sum(Pd_curtailment .+ Qd_curtailment) + sum(P_dg_curtailment .+ Q_dg_curtailment) 
                - 0.01*sum(X_line))

    set_optimizer(model, Gurobi.Optimizer)
    set_optimizer_attributes(model, "OutputFlag" => 0, "LogToConsole" => 0)
    optimize!(model)

    Total_shedded_P = sum(value.(Pd_curtailment))*Sb
    Total_shedded_Q = sum(value.(Qd_curtailment))*Sb
    Total_curtailment_P = sum(value.(P_dg_curtailment))*Sb
    Total_curtailment_Q = sum(value.(Q_dg_curtailment))*Sb
    res_b = value.(b)

    return Total_shedded_P, Total_shedded_Q, Total_curtailment_P, Total_curtailment_Q, res_b
end


sys_file_name = "123Bus_Data.xlsx"
scenario_file_name = "123Bus_Scenario_Data.xlsx"
args = read_data(sys_file_name, scenario_file_name)

Psd = Array{Float64}(undef, args[2], args[1])
Qsd = Array{Float64}(undef, args[2], args[1])
CurtailmentP = Array{Float64}(undef, args[2], args[1])
CurtailmentQ = Array{Float64}(undef, args[2], args[1])
b = Array{Float64}(undef, args[7], args[2], args[1])


# Create a progress bar
p = Progress(args[2]*args[1], 1, "Processing...")

for ind_S in 1:args[2]
    for ind_T in 1:args[1]
        # Psd[ind_S,ind_T], Qsd[ind_S,ind_T], CurtailmentP[ind_S,ind_T], CurtailmentQ[ind_S,ind_T], b[:,ind_S,ind_T] = Deterministic_OneStep(args, ind_S, ind_T)
        _,_,_,_,b[:,ind_S,ind_T] = Deterministic_OneStep(args, ind_S, ind_T)
        next!(p)
    end
    # Update the progress bar
end
npzwrite("b.npy", b)
