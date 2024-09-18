package android.example.clip.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog

private val progressDialogVisibleState = mutableStateOf(false)
private val progressDialogText = mutableStateOf("")

@Composable
fun AppProgressDialog() {
    val isVisible by remember { progressDialogVisibleState }
    if (isVisible) {
        Dialog(onDismissRequest = { /* Progress dialogs are non-cancellable */ }) {
            Box(
                contentAlignment = Alignment.Center,
                modifier =
                Modifier.fillMaxWidth()
                    .background(Color.White, shape = RoundedCornerShape(8.dp))
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    modifier = Modifier.padding(vertical = 24.dp)
                ) {
                    LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    Spacer(modifier = Modifier.padding(4.dp))
                    Text(
                        text = progressDialogText.value,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp)
                    )
                }
            }
        }
    }
}

fun setProgressDialogText(message: String) {
    progressDialogText.value = message
}

fun showProgressDialog() {
    progressDialogVisibleState.value = true
    progressDialogText.value = ""
}

fun hideProgressDialog() {
    progressDialogVisibleState.value = false
}