const enter_btn = document.getElementById("enter-btn")
const credit_policy = document.getElementById('credit_policy')
const int_rate = document.getElementById('int_rate')
const installment = document.getElementById('installment')
const log_annual = document.getElementById('log_annual_inc')
const dti = document.getElementById('dti')
const fico = document.getElementById('fico')
const revol_bal = document.getElementById('revol_bal')
const inq_last_6mths = document.getElementById('inq_last_6mths')
const delinq_2yrs = document.getElementById('delinq_2yrs')
const algo = document.getElementById('algo')
const csrf_token = document.getElementsByName('csrfmiddlewaretoken')[0]
const spiner = document.getElementById('spiner')
const output = document.getElementById("output")

const server_url = `${window.location.protocol}//${window.location.host}`

enter_btn.onclick = async function () {
    credit_policy_value = credit_policy.value
    int_rate_value = int_rate.value
    installment_value = installment.value
    log_annual_value = log_annual.value
    dti_value = dti.value
    fico_value = fico.value
    revol_bal_value = revol_bal.value
    inq_last_6mths_value = inq_last_6mths.value
    delinq_2yrs_value = delinq_2yrs.value
    algo_value = algo.value

    token = csrf_token.value
    spiner.style.display = "flex"

    await $.ajax({
        type: "POST",
        url: server_url,
        headers: {
            "X-CSRFToken": token
        },
        data: {
            "credit_policy": credit_policy_value,
            "int_rate": int_rate_value,
            "installment": installment_value,
            "log_annual_inc": log_annual_value,
            "dti": dti_value,
            "fico": fico_value,
            "revol_bal": revol_bal_value,
            "inq_last_6mths": inq_last_6mths_value,
            "delinq_2yrs": delinq_2yrs_value,
            "algo": algo_value
        },
        success: function (result) {
            output.value = result['answer']
        },
        dataType: "json"
    });
    spiner.style.display = "none"
    output.style.display = "flex"
}