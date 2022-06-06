$(function() {
    $('button#pred').on('click', function(e) {
        console.log('click')
        e.preventDefault()
        var text = $('textarea#paper-text').val()
        var model = $('select#model').val()
        console.log('here')
        console.log(text)
        console.log(model)
        $.ajax({
            url: '/predict',
            type: 'POST',
            data: {
                text: $('textarea#paper-text').val(),
                model: $('select#model').val()
            },
            success: function (response) {
                console.log(response)
                // const result = JSON.parse(response);
                // console.log(result)
                click(response)
            },
            error: function (response) {
            }
        });
    });
});

function click(result) {
    
    console.log('click')
    var ner = result['result']
    document.getElementById("result").style.display = 'block';
    document.getElementById("result").value = ner
    console.log('done')

}