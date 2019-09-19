

shinyServer(function(input, output, session) {
  
  nextwordlist <- reactive({
   (scoreNgrams(input$userInput))
  })
  
  
   output$hist <- renderPlot({
     totalscore <- sum(nextwordlist()[,6])
     outputTabletemp <- nextwordlist()[1:5,]
     # outputTable$Order = (1:10)
     # outputTable <- outputTable[c(2,1)]
     # colnames(outputTable) <- c("Order", "Prediction")
     keeps <- c("nextword","score")
     df <-  outputTabletemp[keeps]
     df$score <- df$score/totalscore
     
     barplot(df$score, names.arg=df$nextword, xlab="Prediction",ylab="Prob",col="blue",main="Top 5 prediction",border="black")
     
     
   }
 
 
  )

})

